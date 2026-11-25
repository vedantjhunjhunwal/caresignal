"""Spark session factory. Local mode for development and CI; on Databricks the
existing session is reused (``SparkSession.getActiveSession()``)."""
from __future__ import annotations

import os

from pyspark.sql import SparkSession


def get_spark(app: str = "caresignal") -> SparkSession:
    active = SparkSession.getActiveSession()
    if active is not None:
        return active
    return (
        SparkSession.builder.appName(app)
        .master(os.environ.get("CARESIGNAL_SPARK_MASTER", "local[2]"))
        .config("spark.driver.memory", os.environ.get("CARESIGNAL_DRIVER_MEMORY", "2g"))
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
