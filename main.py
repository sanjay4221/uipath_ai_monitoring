"""
main.py
───────
Entry point for the UiPath AI Monitoring pipeline.
Usage:
  python main.py              # run full training pipeline
  python main.py --monitor    # demo real-time monitoring on a sample log
"""

import argparse
import sys
import os

# Make src importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from uipath_ai_monitoring.logger import logger
from uipath_ai_monitoring.pipeline.training_pipeline import TrainingPipeline
from dotenv import load_dotenv
load_dotenv()

def run_training():
    """Execute the full ML training pipeline."""
    pipeline = TrainingPipeline(config_path="config/config.yaml")
    metrics = pipeline.run()
    return metrics


def run_monitor_demo():
    """Demo: classify a single synthetic error log entry."""
    from uipath_ai_monitoring.monitoring.log_monitor import UiPathLogMonitor, LogMonitorConfig
    from uipath_ai_monitoring.utils import load_config
    cfg = load_config()

    monitor_cfg = LogMonitorConfig(
        model_dir        = cfg["paths"]["model_dir"],
        groq_model       = cfg["groq"]["model"],
        groq_max_tokens  = cfg["groq"]["max_tokens"],
        groq_temperature = cfg["groq"]["temperature"],
    )
    monitor = UiPathLogMonitor(monitor_cfg)

    sample_log = {
        "timestamp":         "2024-06-15 14:35:22",
        "log_level":         "ERROR",
        "process_name":      "InvoiceProcessing",
        "robot_name":        "Robot_001",
        "message":           "[InvoiceProcessing] SelectorNotFoundException: Element not found on page",
        "exception_type":    "SelectorNotFoundException",
        "exception_message": "Could not find the UI element matching selector '<webctrl tag='BUTTON' aaname='Submit'/>'",
        "duration_ms":       15000,
        "retry_count":       3,
    }

    logger.info("Running real-time monitor demo on sample error log ...")
    result = monitor.monitor(sample_log, use_groq=bool(os.getenv("GROQ_API_KEY")))

    print("\n" + "=" * 60)
    print("🔍  MONITOR RESULT")
    print("=" * 60)
    print(f"  Is Error   : {result['is_error']}")
    print(f"  Confidence : {result['confidence']:.2%}")
    if result["groq_analysis"]:
        print("\n📋  GROQ AI ROOT-CAUSE ANALYSIS:")
        print(result["groq_analysis"])
    else:
        print("\n  (Set GROQ_API_KEY env var to enable AI analysis)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UiPath AI Monitoring Pipeline")
    parser.add_argument("--monitor", action="store_true",
                        help="Run real-time monitoring demo instead of training")
    args = parser.parse_args()

    if args.monitor:
        run_monitor_demo()
    else:
        run_training()
