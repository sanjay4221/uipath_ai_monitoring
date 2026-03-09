"""
data_validation.py
──────────────────
Stage 2 Data Validation
Schema checks, missing-value analysis, log-level sanity, and
data-quality report generation.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from uipath_ai_monitoring.logger import logger
from uipath_ai_monitoring.exception import DataValidationException
from uipath_ai_monitoring.utils import load_config, save_dataframe, save_json


@dataclass
class DataValidationConfig:
    required_columns: List[str]
    valid_log_levels: List[str]
    max_missing_pct: float
    reports_dir: str


@dataclass
class ValidationReport:
    total_rows: int = 0
    total_columns: int = 0
    missing_summary: Dict = field(default_factory=dict)
    schema_valid: bool = False
    log_level_valid: bool = False
    duplicate_rows: int = 0
    error_rate: float = 0.0
    passed: bool = False
    issues: List[str] = field(default_factory=list)


class DataValidation:
    """Validates UiPath log datasets before feature engineering."""

    def __init__(self, config: DataValidationConfig):
        self.config = config
        Path(config.reports_dir).mkdir(parents=True, exist_ok=True)

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        missing_cols = [c for c in self.config.required_columns if c not in df.columns]
        issues = [f"Missing required column: '{c}'" for c in missing_cols]
        return len(missing_cols) == 0, issues

    def _check_missing_values(self, df: pd.DataFrame) -> Tuple[bool, Dict, List[str]]:
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        summary = missing_pct[missing_pct > 0].to_dict()
        issues = []
        passed = True
        for col, pct in summary.items():
            if col in self.config.required_columns and pct > (self.config.max_missing_pct * 100):
                issues.append(f"Column '{col}' has {pct:.1f}% missing (threshold={self.config.max_missing_pct*100:.0f}%)")
                passed = False
        return passed, summary, issues

    def _check_log_levels(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        if "log_level" not in df.columns:
            return False, ["Column 'log_level' absent"]
        df_temp = df.copy()
        df_temp["log_level"] = df_temp["log_level"].str.upper().str.strip()
        valid_set = set(self.config.valid_log_levels)
        found = set(df_temp["log_level"].dropna().unique())
        invalid = found - valid_set
        issues = [f"Invalid log level(s): {invalid}"] if invalid else []
        return len(invalid) == 0, issues

    def _check_duplicates(self, df: pd.DataFrame) -> int:
        return int(df.duplicated().sum())

    def _check_timestamp(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        if "timestamp" not in df.columns:
            return False, ["Column 'timestamp' absent"]
        try:
            pd.to_datetime(df["timestamp"])
            return True, []
        except Exception as e:
            return False, [f"Timestamp parsing failed: {e}"]

    # ── Orchestrator ──────────────────────────────────────────────────────────

    def initiate(self, raw_data_path: str) -> Tuple[str, bool]:
        logger.info("=" * 60)
        logger.info("STAGE 2 ▶ DATA VALIDATION")
        logger.info("=" * 60)

        try:
            df = pd.read_csv(raw_data_path, parse_dates=["timestamp"])
            report = ValidationReport(
                total_rows=len(df),
                total_columns=len(df.columns),
            )

            all_issues: List[str] = []

            # Schema
            ok, issues = self._check_schema(df)
            report.schema_valid = ok
            all_issues.extend(issues)

            # Missing values
            ok, miss_summary, issues = self._check_missing_values(df)
            report.missing_summary = miss_summary
            all_issues.extend(issues)

            # Log levels
            ok, issues = self._check_log_levels(df)
            report.log_level_valid = ok
            all_issues.extend(issues)

            # Duplicates
            report.duplicate_rows = self._check_duplicates(df)
            if report.duplicate_rows > 0:
                logger.warning(f"Found {report.duplicate_rows} duplicate rows — dropping them")
                df = df.drop_duplicates().reset_index(drop=True)

            # Timestamps
            _, issues = self._check_timestamp(df)
            all_issues.extend(issues)

            # Error rate
            if "is_error" in df.columns:
                report.error_rate = round(df["is_error"].mean() * 100, 2)

            report.issues = all_issues
            report.passed = len(all_issues) == 0

            # Persist validated data
            validated_path = raw_data_path.replace("raw_logs", "validated_logs")
            save_dataframe(df, validated_path)

            # Save JSON report
            report_path = str(Path(self.config.reports_dir) / "validation_report.json")
            save_json(report.__dict__, report_path)

            # Print summary
            status = "PASSED" if report.passed else "PASSED WITH WARNINGS"
            logger.info(f"Validation Status : {status}")
            logger.info(f"Total rows        : {report.total_rows}")
            logger.info(f"Duplicate rows    : {report.duplicate_rows}")
            logger.info(f"Error rate        : {report.error_rate}%")
            logger.info(f"Missing summary   : {report.missing_summary}")
            if all_issues:
                for issue in all_issues:
                    logger.warning(f"  ⚠ {issue}")

            return validated_path, report.passed

        except Exception as e:
            raise DataValidationException(str(e))
