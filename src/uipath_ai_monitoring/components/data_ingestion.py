"""
data_ingestion.py
─────────────────
Stage 1 Data Ingestion
Load UiPath log CSV files from disk, generate synthetic data if none exist,
and persist the raw dataset to the artifacts directory.
"""

import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass

from uipath_ai_monitoring.logger import logger
from uipath_ai_monitoring.exception import DataIngestionException
from uipath_ai_monitoring.utils import load_config, save_dataframe


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class DataIngestionConfig:
    raw_data_dir: str
    processed_data_dir: str
    required_columns: list


# ── Synthetic log generator ───────────────────────────────────────────────────

class SyntheticUiPathLogGenerator:
    """
    Generates a realistic synthetic UiPath log dataset for demo / testing.
    In production replace this with your actual CSV loading logic.
    """

    PROCESSES = [
        "InvoiceProcessing", "OrderManagement", "CustomerOnboarding",
        "PayrollAutomation", "DataMigration", "ReportGeneration",
        "EmailProcessing", "DatabaseSync", "ComplianceCheck", "SAPIntegration",
    ]
    ROBOTS = [f"Robot_{i:03d}" for i in range(1, 11)]

    INFO_MESSAGES = [
        "Process started successfully",
        "Reading data from source system",
        "Navigating to target application",
        "Transaction completed successfully",
        "File processed and archived",
        "Data validated and stored",
        "Workflow step completed",
        "Connection established",
        "Authentication successful",
        "Queue item retrieved",
    ]
    ERROR_MESSAGES = [
        "NullReferenceException: Object reference not set",
        "TimeoutException: Operation timed out after 30 seconds",
        "SelectorNotFoundException: Element not found on page",
        "ApplicationException: SAP connection failed",
        "InvalidOperationException: Cannot process empty queue",
        "IOException: File not found or access denied",
        "ArgumentException: Invalid input parameter",
        "WebDriverException: Browser session expired",
        "DatabaseException: Connection pool exhausted",
        "BusinessRuleException: Invoice amount exceeds limit",
    ]
    EXCEPTION_TYPES = [
        "NullReferenceException", "TimeoutException", "SelectorNotFoundException",
        "ApplicationException", "IOException", "ArgumentException",
        "WebDriverException", "DatabaseException", "BusinessRuleException",
        "InvalidOperationException",
    ]

    def generate(self, n_rows: int = 2000, error_rate: float = 0.25) -> pd.DataFrame:
        logger.info(f"Generating {n_rows} synthetic UiPath log rows (error_rate={error_rate})")
        rows = []
        base_time = datetime(2024, 1, 1, 8, 0, 0)

        for i in range(n_rows):
            ts = base_time + timedelta(seconds=random.randint(0, 86400 * 90))
            is_error = random.random() < error_rate
            process = random.choice(self.PROCESSES)
            robot = random.choice(self.ROBOTS)

            if is_error:
                log_level = random.choice(["ERROR", "FATAL", "WARN"])
                message = random.choice(self.ERROR_MESSAGES)
                exc_type = random.choice(self.EXCEPTION_TYPES)
                exc_msg = message
                exc_stack = (
                    f"at UiPath.Core.Activities.{process}.Execute()\n"
                    f"   at System.Activities.WorkflowApplication.Invoke()\n"
                    f"   at {process}.Main()"
                )
            else:
                log_level = random.choice(["INFO", "DEBUG", "TRACE"])
                message = random.choice(self.INFO_MESSAGES)
                exc_type = ""
                exc_msg = ""
                exc_stack = ""

            rows.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "log_level": log_level,
                "process_name": process,
                "robot_name": robot,
                "job_id": f"JOB-{random.randint(10000, 99999)}",
                "message": f"[{process}] {message}",
                "exception_type": exc_type,
                "exception_message": exc_msg,
                "exception_stack": exc_stack,
                "duration_ms": random.randint(50, 5000) if not is_error else random.randint(5000, 30000),
                "retry_count": random.randint(0, 3) if is_error else 0,
                "is_error": int(is_error),
            })

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info(f"Generated dataset: {df.shape}  errors={df['is_error'].sum()}")
        return df


# ── Main component 

class DataIngestion:
    """
    Loads UiPath log CSV files from `raw_data_dir`.
    Falls back to synthetic generation when no CSVs are present.
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config
        Path(config.processed_data_dir).mkdir(parents=True, exist_ok=True)

    def _load_csvs(self, directory: str) -> pd.DataFrame:
        """Load all CSVs from a directory and concatenate them."""
        csv_files = list(Path(directory).glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in '{directory}'. Using synthetic data.")
            return pd.DataFrame()

        dfs = []
        for fp in csv_files:
            logger.info(f"Loading: {fp}")
            df = pd.read_csv(fp)
            df["source_file"] = fp.name
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(csv_files)} file(s) → {combined.shape}")
        return combined

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lower-case and strip column names."""
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

    def initiate(self) -> str:
        """Run ingestion and return path to saved raw dataset."""
        logger.info("=" * 60)
        logger.info("STAGE 1 ▶ DATA INGESTION")
        logger.info("=" * 60)

        try:
            df = self._load_csvs(self.config.raw_data_dir)

            if df.empty:
                gen = SyntheticUiPathLogGenerator()
                df = gen.generate(n_rows=2000)
            else:
                df = self._normalise_columns(df)

            # Basic column presence check
            missing = [c for c in self.config.required_columns if c not in df.columns]
            if missing:
                logger.warning(f"Required columns not found: {missing} — they will be added as empty.")
                for col in missing:
                    df[col] = ""

            output_path = os.path.join(self.config.processed_data_dir, "raw_logs.csv")
            save_dataframe(df, output_path)

            logger.info(f"   Data Ingestion complete → {output_path}")
            logger.info(f"   Shape      : {df.shape}")
            logger.info(f"   Log levels : {df['log_level'].value_counts().to_dict()}")
            logger.info(f"   Date range : {df['timestamp'].min()} → {df['timestamp'].max()}")

            return output_path

        except Exception as e:
            raise DataIngestionException(str(e))
