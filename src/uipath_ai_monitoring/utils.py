"""
utils.py
────────
Shared utility functions for the UiPath AI Monitoring pipeline.
"""

import os
import json
import yaml
import joblib
import pandas as pd
from pathlib import Path
from typing import Any, Dict

from uipath_ai_monitoring.logger import logger
from uipath_ai_monitoring.exception import UiPathAIException


# ── YAML helpers 

def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load YAML configuration file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Config loaded from {config_path}")
        return config
    except Exception as e:
        raise UiPathAIException(f"Failed to load config: {e}")


# ── File I/O helpers

def save_object(obj: Any, filepath: str) -> None:
    """Persist any Python object using joblib."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, filepath)
    logger.info(f"Object saved → {filepath}")


def load_object(filepath: str) -> Any:
    """Load a joblib-serialised object."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"No object found at {filepath}")
    obj = joblib.load(filepath)
    logger.info(f"Object loaded ← {filepath}")
    return obj


def save_json(data: Dict, filepath: str) -> None:
    """Save dictionary as JSON."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4, default=str)
    logger.info(f"JSON saved → {filepath}")


def load_json(filepath: str) -> Dict:
    """Load JSON file as dictionary."""
    with open(filepath, "r") as f:
        return json.load(f)


# ── DataFrame helpers

def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    """Save DataFrame to CSV."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"DataFrame saved → {filepath}  shape={df.shape}")


def load_dataframe(filepath: str) -> pd.DataFrame:
    """Load CSV into DataFrame."""
    df = pd.read_csv(filepath)
    logger.info(f"DataFrame loaded ← {filepath}  shape={df.shape}")
    return df


# ── Validation helpers 

def check_required_columns(df: pd.DataFrame, required: list) -> bool:
    """Return True only if all required columns are present."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(f"Missing columns: {missing}")
        return False
    return True
