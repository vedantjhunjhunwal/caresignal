import pandas as pd
import pytest
from conftest import FIXTURES
from pyspark.sql import functions as F

from caresignal.icd9 import icd9_group, icd9_group_py
from caresignal.pipeline import EXCLUDE_DISPOSITIONS, load_id_maps


def test_quality_rules_quarantine_each_broken_row(lake):
    _, report = lake
    assert report["bronze_rows"] == 2004
    assert report["quarantined_rows"] == 4
    assert report["quarantine_by_rule"] == {
        "gender_known": 1, "los_1_to_14_days": 1, "num_medications_non_negative": 1, "encounter_id_unique": 1}


def test_row_accounting_adds_up(lake):
    _, r = lake
    assert r["bronze_rows"] == r["quarantined_rows"] + r["excluded_death_or_hospice"] + r["silver_rows"]
    assert r["gold_rows"] == r["silver_rows"]


def test_no_raw_identifiers_leave_silver(lake):
    out, _ = lake
    for layer in ("silver/encounters", "gold/readmission_features"):
        cols = pd.read_parquet(out / layer).columns
        assert "patient_nbr" not in cols and "encounter_id" not in cols
    g = pd.read_parquet(out / "gold/readmission_features")
    assert g.patient_key.str.fullmatch(r"[0-9a-f]{64}").all()


def test_splits_are_patient_disjoint(lake):
    out, r = lake
    g = pd.read_parquet(out / "gold/readmission_features")
    assert g.groupby("patient_key")["split"].nunique().max() == 1
    assert r["patients_in_more_than_one_split"] == 0
    assert set(g.split.astype(str)) == {"train", "valid", "test"}


def test_label_and_cohort(lake):
    out, _ = lake
    raw = pd.read_csv(FIXTURES / "diabetic_sample.csv", dtype=str)
    g = pd.read_parquet(out / "gold/readmission_features")
    kept = raw[~raw.discharge_disposition_id.astype(int).isin(EXCLUDE_DISPOSITIONS)]
    assert g.label.sum() <= (kept.readmitted == "<30").sum()  # minus quarantined rows
    assert set(g.label.unique()) <= {0, 1}
    assert "Expired" not in " ".join(g.discharge_disposition_desc.unique())


def test_prior_encounters_has_no_look_ahead(lake):
    out, _ = lake
    g = pd.read_parquet(out / "gold/readmission_features").sort_values("encounter_seq")
    first = g.groupby("patient_key").head(1)
    assert (first.prior_encounters_in_data == 0).all()
    assert (g.groupby("patient_key").prior_encounters_in_data.diff().dropna() == 1).all()


@pytest.mark.parametrize("code,group", [
    ("250.83", "Diabetes"), ("428", "Circulatory"), ("785", "Circulatory"), ("786.05", "Respiratory"),
    ("486", "Respiratory"), ("V45", "Other"), ("E880", "Other"), ("996", "Injury"), ("715", "Musculoskeletal"),
    ("599", "Genitourinary"), ("162", "Neoplasms"), (None, "Missing")])
def test_icd9_reference_codes(code, group):
    assert icd9_group_py(code) == group


def test_spark_icd9_matches_python_on_every_fixture_code(spark):
    raw = pd.read_csv(FIXTURES / "diabetic_sample.csv", dtype=str)
    codes = sorted(set(raw[["diag_1", "diag_2", "diag_3"]].stack()) - {"?"}) + ["V45", "E880"]
    df = spark.createDataFrame([(c,) for c in codes], "code string").withColumn("g", icd9_group(F.col("code")))
    got = {r["code"]: r["g"] for r in df.collect()}
    assert got == {c: icd9_group_py(c) for c in codes}


def test_id_mapping_parses_three_tables():
    m = load_id_maps(FIXTURES / "IDs_mapping.csv")
    assert set(m) == {"admission_type_id", "discharge_disposition_id", "admission_source_id"}
    assert m["admission_type_id"][1] == "Emergency"
    assert m["discharge_disposition_id"][11] == "Expired"
