import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def spark():
    os.environ.setdefault("CARESIGNAL_SPARK_MASTER", "local[1]")
    os.environ.setdefault("CARESIGNAL_DRIVER_MEMORY", "1g")
    from caresignal.spark import get_spark

    s = get_spark("caresignal-tests")
    yield s
    s.stop()


@pytest.fixture(scope="session")
def lake(spark, tmp_path_factory):
    """Run the full pipeline once on the 2,004-row fixture (2,000 real rows + 4 broken)."""
    import shutil

    from caresignal import pipeline

    raw = tmp_path_factory.mktemp("raw")
    shutil.copy(FIXTURES / "diabetic_sample.csv", raw / "diabetic_data.csv")
    shutil.copy(FIXTURES / "IDs_mapping.csv", raw / "IDs_mapping.csv")
    out = tmp_path_factory.mktemp("lake")
    report = pipeline.run(raw, out, salt="test-salt")
    return out, report
