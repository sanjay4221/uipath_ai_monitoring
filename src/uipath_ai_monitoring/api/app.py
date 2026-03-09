"""
app.py
──────
FastAPI REST service exposing:
  POST /predict  classify a single UiPath log entry
  POST /analyse  classify + Groq AI root-cause analysis
  GET  /health   health check
  GET  /metrics  latest model metrics
"""

import os
from dotenv import load_dotenv
load_dotenv()

import json
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from uipath_ai_monitoring.logger import logger
from uipath_ai_monitoring.utils import load_config, load_json
from uipath_ai_monitoring.monitoring.log_monitor import (
    UiPathLogMonitor, LogMonitorConfig,
)

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="UiPath AI Monitoring API",
        description="ML + Groq AI-powered UiPath log error detection service",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    cfg = load_config()
    monitor_cfg = LogMonitorConfig(
        model_dir        = cfg["paths"]["model_dir"],
        groq_model       = cfg["groq"]["model"],
        groq_max_tokens  = cfg["groq"]["max_tokens"],
        groq_temperature = cfg["groq"]["temperature"],
    )
    monitor = UiPathLogMonitor(monitor_cfg)


    class LogEntry(BaseModel):
        timestamp:          Optional[str] = None
        log_level:          str           = "INFO"
        process_name:       str           = "Unknown"
        robot_name:         Optional[str] = None
        job_id:             Optional[str] = None
        message:            str           = ""
        exception_type:     Optional[str] = None
        exception_message:  Optional[str] = None
        duration_ms:        Optional[float] = 0
        retry_count:        Optional[int]   = 0


    @app.get("/health")
    def health():
        return {"status": "healthy", "service": "UiPath AI Monitoring"}


    @app.get("/metrics")
    def get_metrics():
        report_path = cfg["paths"]["reports_dir"] + "/evaluation_report.json"
        if not Path(report_path).exists():
            raise HTTPException(status_code=404, detail="No evaluation report found. Run the training pipeline first.")
        return load_json(report_path)


    @app.post("/predict")
    def predict(entry: LogEntry):
        """Classify a log entry (ML only, no Groq)."""
        result = monitor.monitor(entry.dict(), use_groq=False)
        return {
            "is_error":   result["is_error"],
            "confidence": result["confidence"],
        }


    @app.post("/analyse")
    def analyse(entry: LogEntry):
        """Classify + Groq AI root-cause analysis."""
        result = monitor.monitor(entry.dict(), use_groq=True)
        return result


    if __name__ == "__main__":
        api_cfg = cfg.get("api", {})
        uvicorn.run(
            "uipath_ai_monitoring.api.app:app",
            host   = api_cfg.get("host", "0.0.0.0"),
            port   = api_cfg.get("port", 8000),
            reload = api_cfg.get("reload", True),
        )
else:
    logger.warning("FastAPI not installed API module skipped.")
