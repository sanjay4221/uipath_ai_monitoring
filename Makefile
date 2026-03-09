.PHONY: install train monitor api clean

install:
	pip install -r requirements.txt

train:
	PYTHONPATH=src python main.py

monitor:
	PYTHONPATH=src python main.py --monitor

api:
	cd src && uvicorn uipath_ai_monitoring.api.app:app --reload --port 8000

clean:
	rm -rf artifacts/data/* artifacts/models/* artifacts/reports/* logs/
	@echo "Cleaned artefacts"
