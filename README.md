# UiPath AI Monitoring — ML + Groq AI Pipeline

> An end-to-end Machine Learning and AI-powered system for intelligent UiPath RPA log analysis,
> error classification, and automated root-cause analysis using Groq LLM.

---

## Agenda & Objectives

### Why Analyse UiPath Logs with ML?

UiPath bots generate thousands of log lines per day. Manual monitoring is:
- **Slow** — operators miss errors buried in noise
- **Reactive** — issues are found only after business impact
- **Inconsistent** — different analysts spot different patterns

An ML-powered pipeline solves this by:
1. **Classifying** every log entry as error / non-error in real-time
2. **Predicting** failures before they cascade
3. **Explaining** root causes in plain English via Groq AI (LLaMA-3)

---

## Architecture

Stage 1 Data Ingestion
Stage 2 Data Validation 
Stage 3 Feature Engineering
Stage 5 Model Evaluation
Stage 4 Model Training 
Saved ML Model(best_model.pkl) 
Real-Time Log Monitor and Groq AI Analysis    
FastAPI REST APIpredict /analyse

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Add your UiPath CSV logs
```bash
# Drop CSV files into:
data/sample_logs/your_logs.csv
```
Required columns: `timestamp`, `log_level`, `process_name`, `message`  
Optional: `exception_type`, `exception_message`, `robot_name`, `job_id`

If no CSVs are found, the system auto-generates 2 000 realistic synthetic rows.

### 3. Set Groq API key (for AI analysis)
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```
Get a free key at https://console.groq.com

### 4. Run the training pipeline
```bash
PYTHONPATH=src python main.py
```

### 5. Test real-time monitoring
```bash
PYTHONPATH=src python main.py --monitor
```

### 6. Start the REST API
```bash
cd src
uvicorn uipath_ai_monitoring.api.app:app --reload --port 8000
```

---

## Pipeline Stages

`data_ingestion.py` Load CSVs or generate synthetic UiPath logs
`data_validation.py` Schema check, missing values, log-level audit
`feature_engineering.py` TF-IDF (500 features) + temporal + numeric + text-stat features
`model_trainer.py` Train RandomForest, LogisticRegression, GradientBoosting; pick best by CV F1
`model_evaluation.py` Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix

---

## Groq AI Integration

When an error is detected by the ML model, the system calls **Groq's LLaMA-3-8B** model to generate:

- **Root Cause Analysis** — what went wrong and why
- **Business Impact** — downstream effects on the process
- **Recommended Fix** — concrete remediation steps
- **Prevention Strategy** — how to stop recurrence

---

## Model Artefacts (saved in `artifacts/models/`)

`best_model.pkl` - Trained scikit-learn classifier
`tfidf_vectorizer.pkl` - Fitted TF-IDF transformer
`feature_scaler.pkl` - StandardScaler for numeric features
`level_encoder.pkl` - LabelEncoder for log levels
`process_encoder.pkl` - LabelEncoder for process names
`feature_names.pkl` - List of all feature names

---

## Business Benefits and Description

**Proactive Error Detection** - Classify errors at log ingestion time, before manual review 
**Faster Root-Cause Analysis** - Groq AI explains each failure in seconds 
**Improved Bot Reliability** - Catch failure patterns early; prevent cascading issues
**Audit Trail** - Full JSON reports for compliance and post-mortems
**Easy Integration** - REST API plugs into UiPath Orchestrator webhooks

---

## API Endpoints

---
GET  /health          Health check
GET  /metrics         Latest model evaluation metrics
POST /predict         ML classification only (fast, no LLM)
POST /analyse         ML classification + Groq AI analysis


### Example `/predict` request
json
{
  "timestamp": "2024-06-15 14:35:22",
  "log_level": "ERROR",
  "process_name": "InvoiceProcessing",
  "message": "SelectorNotFoundException: Element not found",
  "exception_type": "SelectorNotFoundException",
  "retry_count": 3
}

## License
MIT — free to use and modify for your RPA monitoring needs.
