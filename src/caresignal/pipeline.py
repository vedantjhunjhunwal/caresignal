"""Bronze -> silver -> gold lakehouse pipeline for 30-day readmission (PySpark).

bronze : raw CSV as strings + ingest metadata (source file, load time, row hash)
silver : typed, '?' -> null, rule-checked; failing rows go to a quarantine table with the
         rule that failed; cohort exclusions (death / hospice discharges) are counted;
         patient and encounter IDs replaced by salted SHA-256 keys (de-identified)
gold   : one row per encounter with model features, label and a patient-disjoint split

    python -m caresignal.pipeline --raw data/raw --lake lakehouse
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from caresignal.icd9 import icd9_group
from caresignal.spark import get_spark

DRUGS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
         "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
         "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
         "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]
INT_COLS = ["encounter_id", "patient_nbr", "admission_type_id", "discharge_disposition_id", "admission_source_id",
            "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
            "number_emergency", "number_inpatient", "number_diagnoses"]
COUNT_COLS = ["num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
              "number_emergency", "number_inpatient", "number_diagnoses"]
# Discharged as expired (11, 19, 20, 21) or to hospice (13, 14): cannot be readmitted, excluded from cohort.
EXCLUDE_DISPOSITIONS = [11, 13, 14, 19, 20, 21]
SPLIT_FRACTIONS = (0.70, 0.85)  # train < 70% <= valid < 85% <= test, by first-encounter order


def bronze(spark: SparkSession, raw: Path) -> DataFrame:
    df = spark.read.csv(str(raw / "diabetic_data.csv"), header=True, inferSchema=False)
    return (df.withColumn("_source_file", F.lit("diabetic_data.csv"))
              .withColumn("_ingested_at", F.current_timestamp())
              .withColumn("_row_hash", F.sha2(F.concat_ws("|", *[F.coalesce(F.col(f"`{c}`"), F.lit(""))
                                                              for c in df.columns]), 256)))


def _rules(df: DataFrame) -> list[tuple[str, F.Column]]:
    """(rule name, condition that must be TRUE for a valid row)."""
    rules = [
        ("encounter_id_present", F.col("encounter_id").isNotNull()),
        ("patient_nbr_present", F.col("patient_nbr").isNotNull()),
        ("gender_known", F.col("gender").isin("Male", "Female")),
        ("los_1_to_14_days", F.col("time_in_hospital").between(1, 14)),
        ("label_valid", F.col("readmitted").isin("NO", ">30", "<30")),
        ("age_band_valid", F.col("age").rlike(r"^\[\d+-\d+\)$")),
    ]
    rules += [(f"{c}_non_negative", F.col(c).isNotNull() & (F.col(c) >= 0)) for c in COUNT_COLS]
    return rules


def silver(bz: DataFrame, ids: dict[str, dict[int, str]], salt: str) -> tuple[DataFrame, DataFrame, dict]:
    df = bz
    for c in bz.columns:
        if not c.startswith("_"):
            df = df.withColumn(c, F.when(F.col(f"`{c}`") == "?", None).otherwise(F.col(f"`{c}`")))
    for c in INT_COLS:
        df = df.withColumn(c, F.col(c).cast("int"))

    dup = Window.partitionBy("encounter_id").orderBy("_row_hash")
    df = df.withColumn("_dup_rank", F.row_number().over(dup))
    rules = _rules(df) + [("encounter_id_unique", F.col("_dup_rank") == 1)]
    failed = F.array_compact(F.array(*[F.when(~F.coalesce(cond, F.lit(False)), F.lit(name)) for name, cond in rules]))
    df = df.withColumn("_failed_rules", failed)
    quarantine = df.filter(F.size("_failed_rules") > 0)
    valid = df.filter(F.size("_failed_rules") == 0)

    excluded = valid.filter(F.col("discharge_disposition_id").isin(EXCLUDE_DISPOSITIONS))
    cohort = valid.filter(~F.col("discharge_disposition_id").isin(EXCLUDE_DISPOSITIONS))

    for col, mapping in ids.items():  # decode admission/discharge/source IDs to labels
        m = F.create_map(*[x for k, v in mapping.items() for x in (F.lit(k), F.lit(v))])
        cohort = cohort.withColumn(col.replace("_id", "_desc"), F.coalesce(m[F.col(col)], F.lit("Unknown")))

    # De-identification: salted hashes replace raw IDs; raw IDs never leave silver's input.
    cohort = (cohort
              .withColumn("patient_key", F.sha2(F.concat(F.lit(salt), F.col("patient_nbr").cast("string")), 256))
              .withColumn("encounter_key", F.sha2(F.concat(F.lit(salt), F.col("encounter_id").cast("string")), 256))
              .withColumn("encounter_seq", F.dense_rank().over(Window.orderBy("encounter_id")))  # order only
              .drop("patient_nbr", "encounter_id", "_dup_rank", "_failed_rules", "weight"))

    rule_counts = (quarantine.select(F.explode("_failed_rules").alias("rule")).groupBy("rule").count().collect())
    report = {
        "bronze_rows": bz.count(),
        "quarantined_rows": quarantine.count(),
        "quarantine_by_rule": {r["rule"]: r["count"] for r in rule_counts},
        "excluded_death_or_hospice": excluded.count(),
        "silver_rows": cohort.count(),
    }
    return cohort, quarantine.drop("_dup_rank"), report


def gold(sv: DataFrame) -> DataFrame:
    drug_changed = sum(F.when(F.col(f"`{d}`").isin("Up", "Down"), 1).otherwise(0) for d in DRUGS)
    drug_on = sum(F.when(F.col(f"`{d}`") != "No", 1).otherwise(0) for d in DRUGS)
    age_lo = F.regexp_extract("age", r"\[(\d+)-", 1).cast("int")
    pw = Window.partitionBy("patient_key").orderBy("encounter_seq")
    top_spec = ["InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice", "Cardiology", "Surgery-General",
                "Nephrology", "Orthopedics", "Orthopedics-Reconstructive", "Radiologist", "Pulmonology"]
    g = (sv
         .withColumn("age_years", age_lo + 5)
         .withColumn("age_band", F.when(age_lo < 40, "<40").when(age_lo < 60, "40-59")
                     .when(age_lo < 80, "60-79").otherwise("80+"))
         .withColumn("is_male", (F.col("gender") == "Male").cast("int"))
         .withColumn("diag_1_grp", icd9_group(F.col("diag_1")))
         .withColumn("diag_2_grp", icd9_group(F.col("diag_2")))
         .withColumn("diag_3_grp", icd9_group(F.col("diag_3")))
         .withColumn("n_drugs_changed", drug_changed)
         .withColumn("n_drugs_on", drug_on)
         .withColumn("prior_visits", F.col("number_outpatient") + F.col("number_emergency") + F.col("number_inpatient"))
         # encounters of the same patient seen *before* this one in the data: no look-ahead
         .withColumn("prior_encounters_in_data", F.row_number().over(pw) - 1)
         .withColumn("specialty", F.when(F.col("medical_specialty").isin(top_spec), F.col("medical_specialty"))
                     .when(F.col("medical_specialty").isNull(), "Missing").otherwise("Other"))
         .withColumn("payer_missing", F.col("payer_code").isNull().cast("int"))
         .withColumn("a1c", F.coalesce("A1Cresult", F.lit("None")))
         .withColumn("max_glu", F.coalesce("max_glu_serum", F.lit("None")))
         .withColumn("race_audit", F.coalesce("race", F.lit("Unknown")))  # audit only, never a model feature
         .withColumn("label", (F.col("readmitted") == "<30").cast("int")))

    first = g.groupBy("patient_key").agg(F.min("encounter_seq").alias("first_seq"))
    n = g.agg(F.max("encounter_seq")).first()[0]
    lo, hi = int(n * SPLIT_FRACTIONS[0]), int(n * SPLIT_FRACTIONS[1])
    g = (g.join(first, "patient_key")
          .withColumn("split", F.when(F.col("first_seq") <= lo, "train")
                      .when(F.col("first_seq") <= hi, "valid").otherwise("test"))
          .drop("first_seq"))
    keep = ["encounter_key", "patient_key", "encounter_seq", "split", "label", "race_audit", "age_band",
            "age_years", "is_male", "admission_type_desc", "discharge_disposition_desc", "admission_source_desc",
            "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
            "number_emergency", "number_inpatient", "prior_visits", "prior_encounters_in_data", "number_diagnoses",
            "diag_1_grp", "diag_2_grp", "diag_3_grp", "a1c", "max_glu", "insulin", "change", "diabetesMed",
            "n_drugs_changed", "n_drugs_on", "specialty", "payer_missing"]
    return g.select(*keep)


def load_id_maps(path: Path) -> dict[str, dict[int, str]]:
    """IDs_mapping.csv stacks three small tables, each starting with its own header row."""
    maps, cur = {}, None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.strip(",") == "":
            continue
        k, v = line.split(",", 1)
        if k.endswith("_id"):
            cur = k
            maps[cur] = {}
        elif cur:
            maps[cur][int(k)] = v.strip().strip('"') or "Unknown"
    return maps


def run(raw: Path, lake: Path, salt: str | None = None) -> dict:
    spark = get_spark()
    salt = salt or os.environ.get("CARESIGNAL_SALT", "dev-only-salt-change-me")
    bz = bronze(spark, raw)
    bz.write.mode("overwrite").parquet(str(lake / "bronze/encounters"))
    bz = spark.read.parquet(str(lake / "bronze/encounters"))
    sv, qt, report = silver(bz, load_id_maps(raw / "IDs_mapping.csv"), salt)
    sv.write.mode("overwrite").parquet(str(lake / "silver/encounters"))
    qt.write.mode("overwrite").parquet(str(lake / "silver/quarantine"))
    gd = gold(spark.read.parquet(str(lake / "silver/encounters")))
    gd.write.mode("overwrite").partitionBy("split").parquet(str(lake / "gold/readmission_features"))
    gd = spark.read.parquet(str(lake / "gold/readmission_features"))
    by_split = {r["split"]: {"rows": r["n"], "patients": r["p"], "readmit_30_rate": round(r["rate"], 4)}
                for r in gd.groupBy("split").agg(F.count("*").alias("n"), F.countDistinct("patient_key").alias("p"),
                                                 F.avg("label").alias("rate")).collect()}
    overlap = (gd.select("patient_key", "split").distinct().groupBy("patient_key").count().filter("count > 1").count())
    report.update({"gold_rows": gd.count(), "splits": by_split, "patients_in_more_than_one_split": overlap})
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", type=Path)
    ap.add_argument("--lake", default="lakehouse", type=Path)
    a = ap.parse_args()
    report = run(a.raw, a.lake)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_quality.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
