"""
log_monitor.py
──────────────
Real-time log monitoring + Groq AI root-cause analysis.
Loads a trained model and classifies incoming UiPath log entries,
then invokes Groq LLM for natural-language explanations.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

from uipath_ai_monitoring.logger import logger
from uipath_ai_monitoring.exception import GroqAIException
from uipath_ai_monitoring.utils import load_object
from scipy.sparse import hstack, csr_matrix


@dataclass
class LogMonitorConfig:
    model_dir: str
    groq_model: str = "llama3-8b-8192"
    groq_max_tokens: int = 1024
    groq_temperature: float = 0.2


class GroqAIAnalyzer:
    """Wraps the Groq client for root-cause analysis of UiPath errors."""

    SYSTEM_PROMPT = """You are an expert RPA (Robotic Process Automation) engineer 
specialising in UiPath automation. Analyse the provided UiPath log entry and give:
1. Root Cause Analysis (2–3 sentences)
2. Likely Business Impact
3. Recommended Fix (concrete, actionable steps)
4. Prevention Strategy

Be concise, technical, and practical. Format with clear headings."""

    def __init__(self, config: LogMonitorConfig):
        self.config = config
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise GroqAIException("GROQ_API_KEY environment variable not set.")
                self._client = Groq(api_key=api_key)
            except ImportError:
                raise GroqAIException("groq package not installed. Run: pip install groq")
        return self._client

    def analyse(self, log_entry: Dict) -> str:
        """Send a log entry to Groq and return the AI analysis."""
        client = self._get_client()
        prompt = f"""Analyse this UiPath automation log entry:

Process     : {log_entry.get('process_name', 'Unknown')}
Log Level   : {log_entry.get('log_level', 'Unknown')}
Message     : {log_entry.get('message', '')}
Exception   : {log_entry.get('exception_type', 'None')}
Exc. Detail : {log_entry.get('exception_message', 'None')}
Timestamp   : {log_entry.get('timestamp', 'Unknown')}
Robot       : {log_entry.get('robot_name', 'Unknown')}
Retry Count : {log_entry.get('retry_count', 0)}
"""
        try:
            response = client.chat.completions.create(
                model=self.config.groq_model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=self.config.groq_max_tokens,
                temperature=self.config.groq_temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise GroqAIException(f"Groq API call failed: {e}")


class UiPathLogMonitor:
    """
    Real-time monitor that:
      1. Accepts a raw log entry dict
      2. Classifies it (error / non-error) using the trained ML model
      3. If classified as error → requests Groq AI root-cause analysis
    """

    def __init__(self, config: LogMonitorConfig):
        self.config = config
        self.groq = GroqAIAnalyzer(config)
        self._model      = None
        self._tfidf      = None
        self._scaler     = None
        self._l_encoder  = None
        self._p_encoder  = None

    def _load_artifacts(self):
        if self._model is None:
            mdir = self.config.model_dir
            self._model     = load_object(f"{mdir}/best_model.pkl")
            self._tfidf     = load_object(f"{mdir}/tfidf_vectorizer.pkl")
            self._scaler    = load_object(f"{mdir}/feature_scaler.pkl")
            self._l_encoder = load_object(f"{mdir}/level_encoder.pkl")
            self._p_encoder = load_object(f"{mdir}/process_encoder.pkl")
            logger.info("Monitor: ML artefacts loaded")

    def _featurise(self, entry: Dict) -> "csr_matrix":
        """Convert a single log entry dict to a feature vector."""
        from scipy.sparse import hstack, csr_matrix
        import pandas as pd
        import re

        # Text
        text = " ".join([
            str(entry.get("message", "")),
            str(entry.get("exception_type", "")),
            str(entry.get("exception_message", "")),
        ]).lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        tfidf_vec = self._tfidf.transform([text])

        # Structured
        row = {}
        ts = pd.to_datetime(entry.get("timestamp"), errors="coerce")
        if ts is not None and not pd.isnull(ts):
            row["hour"]              = ts.hour
            row["day_of_week"]       = ts.dayofweek
            row["is_weekend"]        = int(ts.dayofweek >= 5)
            row["is_business_hours"] = int(8 <= ts.hour <= 18 and ts.dayofweek < 5)
        else:
            row.update({"hour": 0, "day_of_week": 0, "is_weekend": 0, "is_business_hours": 0})

        lvl = str(entry.get("log_level", "INFO")).upper()
        try:
            row["log_level_encoded"] = int(self._l_encoder.transform([lvl])[0])
        except Exception:
            row["log_level_encoded"] = 0
        row["is_error_level"] = int(lvl in ["ERROR", "FATAL"])
        row["is_warn_level"]  = int(lvl in ["WARN", "WARNING"])

        proc = str(entry.get("process_name", "Unknown"))
        try:
            row["process_encoded"] = int(self._p_encoder.transform([proc])[0])
        except Exception:
            row["process_encoded"] = 0

        dur = float(entry.get("duration_ms", 0))
        row["duration_ms"]     = dur
        row["log_duration_ms"] = np.log1p(dur)
        row["retry_count"]     = float(entry.get("retry_count", 0))

        msg = str(entry.get("message", ""))
        row["msg_length"]     = len(msg)
        row["msg_word_count"] = len(msg.split())
        kw = ["exception", "error", "timeout", "failed", "null", "invalid"]
        row["has_exception_kw"] = int(any(k in msg.lower() for k in kw))
        row["has_exception"]    = int(bool(entry.get("exception_type")))

        structured_df  = pd.DataFrame([row])
        structured_arr = self._scaler.transform(structured_df)
        structured_sp  = csr_matrix(structured_arr)

        return hstack([tfidf_vec, structured_sp])

    def monitor(self, log_entry: Dict, use_groq: bool = True) -> Dict:
        """
        Classify a single log entry and optionally invoke Groq AI.

        Returns:
            {
              "is_error": bool,
              "confidence": float,
              "groq_analysis": str | None,
            }
        """
        self._load_artifacts()
        X = self._featurise(log_entry)

        pred = self._model.predict(X)[0]
        confidence = 0.0
        if hasattr(self._model, "predict_proba"):
            confidence = float(self._model.predict_proba(X)[0][1])

        result = {
            "is_error":    bool(pred),
            "confidence":  round(confidence, 4),
            "groq_analysis": None,
        }

        if pred and use_groq:
            logger.info(f"Error detected (confidence={confidence:.2%}) – calling Groq AI ...")
            try:
                result["groq_analysis"] = self.groq.analyse(log_entry)
            except GroqAIException as e:
                result["groq_analysis"] = f"[Groq unavailable: {e}]"

        return result
