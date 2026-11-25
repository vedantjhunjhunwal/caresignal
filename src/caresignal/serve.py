"""FastAPI scoring service for the registered readmission model.

Loads ``models:/caresignal-readmission@champion`` from the MLflow registry, so promoting a
new version is an alias change, not a redeploy. The request schema only has de-identified
clinical features: unknown fields (names, MRNs, addresses...) are rejected with 422.

    MLFLOW_TRACKING_URI=sqlite:///mlflow.db uvicorn caresignal.serve:app
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

MODEL_URI = os.environ.get("CARESIGNAL_MODEL_URI", "models:/caresignal-readmission@champion")
log = logging.getLogger("caresignal.serve")


class Encounter(BaseModel):
    model_config = ConfigDict(extra="forbid")  # anything outside the schema (e.g. PHI) is refused

    age_years: int = Field(ge=0, le=110)
    is_male: Literal[0, 1]
    time_in_hospital: int = Field(ge=1, le=14)
    num_lab_procedures: int = Field(ge=0)
    num_procedures: int = Field(ge=0)
    num_medications: int = Field(ge=0)
    number_outpatient: int = Field(ge=0)
    number_emergency: int = Field(ge=0)
    number_inpatient: int = Field(ge=0)
    prior_visits: int = Field(ge=0)
    prior_encounters_in_data: int = Field(ge=0)
    number_diagnoses: int = Field(ge=1)
    n_drugs_changed: int = Field(ge=0)
    n_drugs_on: int = Field(ge=0)
    payer_missing: Literal[0, 1]
    admission_type_desc: str
    discharge_disposition_desc: str
    admission_source_desc: str
    diag_1_grp: str
    diag_2_grp: str
    diag_3_grp: str
    a1c: str
    max_glu: str
    insulin: str
    change: str
    diabetesMed: str
    specialty: str


class Score(BaseModel):
    risk_30d: float
    model_version: str


@lru_cache(maxsize=1)
def model():
    m = mlflow.pyfunc.load_model(MODEL_URI)
    return m, m.metadata.run_id


app = FastAPI(title="CareSignal readmission risk", version="1.0")


@app.get("/health")
def health() -> dict:
    try:
        _, run = model()
        return {"status": "ok", "model_uri": MODEL_URI, "run_id": run}
    except Exception as e:  # registry unreachable or alias missing
        raise HTTPException(503, f"model not loaded: {type(e).__name__}") from e


@app.post("/score", response_model=list[Score])
def score(batch: list[Encounter]) -> list[Score]:
    if not batch or len(batch) > 1000:
        raise HTTPException(422, "send 1-1000 encounters")
    m, run = model()
    p = m.predict(pd.DataFrame([e.model_dump() for e in batch]))
    log.info("scored %d encounters", len(batch))  # counts only: no feature values in logs
    return [Score(risk_30d=round(float(x), 4), model_version=run) for x in p]
