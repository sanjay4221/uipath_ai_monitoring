import os
import sys
sys.path.insert(0, "src")

from dotenv import load_dotenv
load_dotenv()
from uipath_ai_monitoring.monitoring.log_monitor import UiPathLogMonitor, LogMonitorConfig
from uipath_ai_monitoring.utils import load_config

# Confirm key is loaded
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("ERROR: GROQ_API_KEY not found in .env")
    sys.exit(1)
else:
    print(f"OK: GROQ_API_KEY loaded ({api_key[:8]}...)")



cfg = load_config()

monitor_cfg = LogMonitorConfig(
    model_dir        = cfg["paths"]["model_dir"],
    groq_model       = cfg["groq"]["model"],
    groq_max_tokens  = cfg["groq"]["max_tokens"],
    groq_temperature = cfg["groq"]["temperature"],
)

monitor = UiPathLogMonitor(monitor_cfg)

# ── SAP Login Failure log entry 
sap_log = {
    "timestamp":         "2024-06-15 14:35:22",
    "log_level":         "ERROR",
    "process_name":      "SAPIntegration",
    "robot_name":        "Robot_001",
    "message":           "[SAPIntegration] ApplicationException: SAP login failed - invalid credentials or system unavailable",
    "exception_type":    "ApplicationException",
    "exception_message": "SAP system R3 connection failed. Login rejected for user SVCACCOUNT. Error: RFC_LOGON_FAILURE",
    "duration_ms":       25000,
    "retry_count":       3,
}

print("\n" + "="*60)
print("INPUT LOG ENTRY")
print("="*60)
for k, v in sap_log.items():
    print(f"  {k:<22}: {v}")

print("\n" + "="*60)
print("RUNNING ML CLASSIFIER + GROQ AI RCA")
print("="*60)

result = monitor.monitor(sap_log, use_groq=True)

print(f"\n  Is Error    : {result['is_error']}")
print(f"  Confidence  : {result['confidence']:.2%}")

if result["groq_analysis"]:
    print("\n" + "="*60)
    print(" GROQ AI ROOT-CAUSE ANALYSIS")
    print("="*60)
    print(result["groq_analysis"])
else:
    print("\n Groq analysis not available — check GROQ_API_KEY in .env")

print("="*60)