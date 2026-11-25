"""ICD-9 diagnosis grouping used by Strack et al. (2014), the paper that released the
Diabetes 130-US Hospitals data. Returned as a Spark Column so it runs inside the pipeline."""
from __future__ import annotations

from pyspark.sql import Column
from pyspark.sql import functions as F

GROUPS = ["Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
          "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing"]


def icd9_group(col: Column) -> Column:
    code = F.trim(col)
    num = code.try_cast("double")  # V/E codes and nulls become null here
    return (
        F.when(code.isNull(), "Missing")
        .when(code.startswith("250"), "Diabetes")
        .when(num.isNull(), "Other")
        .when(((num >= 390) & (num < 460)) | (F.floor(num) == 785), "Circulatory")
        .when(((num >= 460) & (num < 520)) | (F.floor(num) == 786), "Respiratory")
        .when(((num >= 520) & (num < 580)) | (F.floor(num) == 787), "Digestive")
        .when((num >= 800) & (num < 1000), "Injury")
        .when((num >= 710) & (num < 740), "Musculoskeletal")
        .when(((num >= 580) & (num < 630)) | (F.floor(num) == 788), "Genitourinary")
        .when((num >= 140) & (num < 240), "Neoplasms")
        .otherwise("Other")
    )


def icd9_group_py(code: str | None) -> str:
    """Plain-Python twin of :func:`icd9_group`, used by tests to cross-check the Spark logic."""
    if code is None or code.strip() in ("", "?"):
        return "Missing"
    code = code.strip()
    if code.startswith("250"):
        return "Diabetes"
    try:
        n = float(code)
    except ValueError:
        return "Other"
    f = int(n // 1)
    if 390 <= n < 460 or f == 785:
        return "Circulatory"
    if 460 <= n < 520 or f == 786:
        return "Respiratory"
    if 520 <= n < 580 or f == 787:
        return "Digestive"
    if 800 <= n < 1000:
        return "Injury"
    if 710 <= n < 740:
        return "Musculoskeletal"
    if 580 <= n < 630 or f == 788:
        return "Genitourinary"
    if 140 <= n < 240:
        return "Neoplasms"
    return "Other"
