"""
feature_engineering.py
──────────────────────
Stage 3 Feature Engineering
Transforms raw UiPath log text + metadata into ML-ready features:
  • TF-IDF on combined text (message + exception)
  • Categorical encoding of log_level / process_name
  • Temporal features from timestamp
  • Numeric features (duration, retry_count)
"""

import re
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.sparse import hstack, csr_matrix

from uipath_ai_monitoring.logger import logger
from uipath_ai_monitoring.exception import FeatureEngineeringException
from uipath_ai_monitoring.utils import save_dataframe, save_object


@dataclass
class FeatureEngineeringConfig:
    tfidf_max_features: int
    ngram_range: tuple
    min_df: int
    target_column: str
    processed_data_dir: str
    model_dir: str


class TextCleaner:
    """Light NLP cleaner for UiPath log messages."""

    EXCEPTION_KEYWORDS = [
        "exception", "error", "timeout", "failed", "null",
        "invalid", "not found", "denied", "expired", "exhausted",
    ]

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"\b[a-f0-9]{8,}\b", " HEXID ", text)        # hex IDs
        text = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", " DATETIME ", text)
        text = re.sub(r"\b\d+\b", " NUM ", text)                     # integers
        text = re.sub(r"[^a-z\s]", " ", text)                        # punctuation
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def has_exception_keyword(self, text: str) -> int:
        if not isinstance(text, str):
            return 0
        low = text.lower()
        return int(any(k in low for k in self.EXCEPTION_KEYWORDS))


class FeatureEngineering:
    """Build ML feature matrix from validated UiPath log DataFrame."""

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.cleaner = TextCleaner()
        self.tfidf = TfidfVectorizer(
            max_features=config.tfidf_max_features,
            ngram_range=config.ngram_range,
            min_df=config.min_df,
            sublinear_tf=True,
        )
        self.scaler = StandardScaler()
        self.level_encoder = LabelEncoder()
        self.process_encoder = LabelEncoder()
        Path(config.processed_data_dir).mkdir(parents=True, exist_ok=True)
        Path(config.model_dir).mkdir(parents=True, exist_ok=True)

    # ── Helper builders ───────────────────────────────────────────────────────

    def _build_text_column(self, df: pd.DataFrame) -> pd.Series:
        """Combine message + exception fields into one clean text string."""
        cols = []
        for col in ["message", "exception_type", "exception_message"]:
            if col in df.columns:
                cols.append(df[col].fillna("").astype(str))
        combined = pd.Series([""] * len(df))
        for c in cols:
            combined = combined + " " + c
        return combined.apply(self.cleaner.clean)

    def _temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract hour-of-day, day-of-week, and is_business_hours flags."""
        feats = pd.DataFrame()
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            feats["hour"] = ts.dt.hour.fillna(0)
            feats["day_of_week"] = ts.dt.dayofweek.fillna(0)
            feats["is_weekend"] = (feats["day_of_week"] >= 5).astype(int)
            feats["is_business_hours"] = (
                (feats["hour"] >= 8) & (feats["hour"] <= 18) & (~feats["is_weekend"].astype(bool))
            ).astype(int)
        return feats

    def _log_level_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame()
        if "log_level" in df.columns:
            lvl = df["log_level"].str.upper().str.strip().fillna("INFO")
            feats["log_level_encoded"] = self.level_encoder.fit_transform(lvl)
            feats["is_error_level"] = lvl.isin(["ERROR", "FATAL"]).astype(int)
            feats["is_warn_level"] = lvl.isin(["WARN", "WARNING"]).astype(int)
        return feats

    def _process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame()
        if "process_name" in df.columns:
            pn = df["process_name"].fillna("Unknown")
            feats["process_encoded"] = self.process_encoder.fit_transform(pn)
        return feats

    def _numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame()
        if "duration_ms" in df.columns:
            feats["duration_ms"] = df["duration_ms"].fillna(0).astype(float)
            feats["log_duration_ms"] = np.log1p(feats["duration_ms"])
        if "retry_count" in df.columns:
            feats["retry_count"] = df["retry_count"].fillna(0).astype(float)
        return feats

    def _text_stat_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame()
        if "message" in df.columns:
            msg = df["message"].fillna("")
            feats["msg_length"] = msg.str.len()
            feats["msg_word_count"] = msg.str.split().str.len()
            feats["has_exception_kw"] = msg.apply(self.cleaner.has_exception_keyword)
        if "exception_type" in df.columns:
            feats["has_exception"] = df["exception_type"].notna().astype(int)
        return feats

    # ── Public API ────────────────────────────────────────────────────────────

    def initiate(self, validated_data_path: str) -> Tuple[str, str]:
        logger.info("=" * 60)
        logger.info("STAGE 3 ▶ FEATURE ENGINEERING")
        logger.info("=" * 60)

        try:
            df = pd.read_csv(validated_data_path, parse_dates=["timestamp"])
            logger.info(f"Loaded {df.shape} rows for feature engineering")

            # 1. Text features via TF-IDF
            text_series = self._build_text_column(df)
            tfidf_matrix = self.tfidf.fit_transform(text_series)
            logger.info(f"TF-IDF matrix: {tfidf_matrix.shape}")

            # 2. Structured features
            frames = [
                self._temporal_features(df),
                self._log_level_features(df),
                self._process_features(df),
                self._numeric_features(df),
                self._text_stat_features(df),
            ]
            structured = pd.concat([f for f in frames if not f.empty], axis=1)
            logger.info(f"Structured features: {structured.shape}")

            # Scale numeric structured features
            structured_scaled = self.scaler.fit_transform(structured)
            structured_sparse = csr_matrix(structured_scaled)

            # 3. Combine TF-IDF + structured → final feature matrix
            X = hstack([tfidf_matrix, structured_sparse])
            logger.info(f"Final feature matrix: {X.shape}")

            # 4. Target vector
            if self.config.target_column in df.columns:
                y = df[self.config.target_column].values
            else:
                # Fallback: derive from log_level
                logger.warning(f"Target column '{self.config.target_column}' not found — deriving from log_level")
                y = df["log_level"].str.upper().isin(["ERROR", "FATAL"]).astype(int).values

            logger.info(f"Target distribution: errors={y.sum()}, non-errors={len(y)-y.sum()}")

            # 5. Save artefacts
            feature_names = (
                [f"tfidf_{t}" for t in self.tfidf.get_feature_names_out()]
                + list(structured.columns)
            )

            # Save sparse matrix as npz
            from scipy.sparse import save_npz
            X_path = str(Path(self.config.processed_data_dir) / "features.npz")
            save_npz(X_path, X)

            # Save target
            y_path = str(Path(self.config.processed_data_dir) / "target.npy")
            np.save(y_path, y)

            # Save transformer objects
            save_object(self.tfidf,              str(Path(self.config.model_dir) / "tfidf_vectorizer.pkl"))
            save_object(self.scaler,             str(Path(self.config.model_dir) / "feature_scaler.pkl"))
            save_object(self.level_encoder,      str(Path(self.config.model_dir) / "level_encoder.pkl"))
            save_object(self.process_encoder,    str(Path(self.config.model_dir) / "process_encoder.pkl"))
            save_object(feature_names,           str(Path(self.config.model_dir) / "feature_names.pkl"))

            logger.info(f"   Feature Engineering complete")
            logger.info(f"   Feature matrix : {X.shape}")
            logger.info(f"   Target shape   : {y.shape}")

            return X_path, y_path

        except Exception as e:
            raise FeatureEngineeringException(str(e))
