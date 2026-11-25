import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.isotonic import IsotonicRegression

from caresignal import serve
from caresignal.train import CATEGORICAL, FEATURES, NUMERIC, ReadmissionModel

ENC = {
    "age_years": 75, "is_male": 1, "time_in_hospital": 4, "num_lab_procedures": 45, "num_procedures": 1,
    "num_medications": 16, "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 2,
    "prior_visits": 2, "prior_encounters_in_data": 1, "number_diagnoses": 9, "n_drugs_changed": 1, "n_drugs_on": 2,
    "payer_missing": 0, "admission_type_desc": "Emergency", "discharge_disposition_desc": "Discharged to home",
    "admission_source_desc": " Emergency Room", "diag_1_grp": "Circulatory", "diag_2_grp": "Diabetes",
    "diag_3_grp": "Other", "a1c": "None", "max_glu": "None", "insulin": "Steady", "change": "Ch", "diabetesMed": "Yes",
    "specialty": "InternalMedicine",
}


@pytest.fixture(scope="module")
def tiny_model():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({c: rng.integers(0, 5, n) for c in NUMERIC})
    levels = {c: ["a", "b"] for c in CATEGORICAL}
    for c in CATEGORICAL:
        df[c] = rng.choice(["a", "b"], n)
    y = (df.number_inpatient + rng.normal(0, 1, n) > 3).astype(int)
    m = ReadmissionModel(None, None, levels)
    booster = lgb.train({"objective": "binary", "verbose": -1, "min_data_in_leaf": 20}, lgb.Dataset(m.prep(df), y), 30)
    iso = IsotonicRegression(out_of_bounds="clip").fit(booster.predict(m.prep(df)), y)
    return ReadmissionModel(booster, iso, levels)


def test_model_outputs_probabilities_and_tolerates_unseen_categories(tiny_model):
    df = pd.DataFrame([ENC])  # every category here is unseen by the tiny model
    p = tiny_model.predict(None, df)
    assert p.shape == (1,) and 0 <= p[0] <= 1
    assert tiny_model.prep(df)[CATEGORICAL].isna().all(axis=None)


def test_race_is_never_a_feature():
    assert "race_audit" not in FEATURES and "race" not in FEATURES


@pytest.fixture
def client(monkeypatch, tiny_model):
    class Wrapped:
        def predict(self, df):
            return tiny_model.predict(None, df)

    monkeypatch.setattr(serve, "model", lambda: (Wrapped(), "test-run"))
    return TestClient(serve.app)


def test_score_ok(client):
    r = client.post("/score", json=[ENC, {**ENC, "number_inpatient": 4}])
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2 and all(0 <= b["risk_30d"] <= 1 for b in body)
    assert body[0]["model_version"] == "test-run"


def test_phi_fields_are_rejected(client):
    r = client.post("/score", json=[{**ENC, "patient_name": "Jane Doe", "mrn": "12345"}])
    assert r.status_code == 422


@pytest.mark.parametrize("field,value", [("time_in_hospital", 0), ("time_in_hospital", 30), ("age_years", -1)])
def test_out_of_range_values_are_rejected(client, field, value):
    assert client.post("/score", json=[{**ENC, field: value}]).status_code == 422


def test_empty_batch_rejected(client):
    assert client.post("/score", json=[]).status_code == 422
