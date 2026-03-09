"""
model_evaluation.py
───────────────────
Stage 5 Model Evaluation
Computes accuracy, precision, recall, F1, ROC-AUC, and
generates a confusion-matrix plot + full JSON report.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)

from uipath_ai_monitoring.logger import logger
from uipath_ai_monitoring.exception import ModelEvaluationException
from uipath_ai_monitoring.utils import load_object, save_json


@dataclass
class ModelEvaluationConfig:
    reports_dir: str
    model_dir: str


class ModelEvaluation:
    """Evaluate the trained model and produce metrics + visualisations."""

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        Path(config.reports_dir).mkdir(parents=True, exist_ok=True)

    def _compute_metrics(self, y_true, y_pred, y_prob=None) -> dict:
        metrics = {
            "accuracy":  float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score":  float(f1_score(y_true, y_pred, zero_division=0)),
        }
        if y_prob is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            except Exception:
                metrics["roc_auc"] = None
        metrics["classification_report"] = classification_report(
            y_true, y_pred, target_names=["Non-Error", "Error"]
        )
        return metrics

    def _plot_confusion_matrix(self, y_true, y_pred) -> str:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["Non-Error", "Error"])
        disp.plot(ax=ax, colorbar=True, cmap="Blues")
        ax.set_title("Confusion Matrix – UiPath Error Classifier", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = str(Path(self.config.reports_dir) / "confusion_matrix.png")
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Confusion matrix saved → {path}")
        return path

    def _plot_metrics_bar(self, metrics: dict) -> str:
        keys   = ["accuracy", "precision", "recall", "f1_score"]
        values = [metrics.get(k, 0) for k in keys]
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(keys, values, color=colors, width=0.5, edgecolor="white")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Model Performance Metrics", fontsize=13, fontweight="bold")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=11)
        if metrics.get("roc_auc"):
            ax.axhline(metrics["roc_auc"], color="purple", linestyle="--",
                       label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
            ax.legend()
        plt.tight_layout()
        path = str(Path(self.config.reports_dir) / "metrics_bar.png")
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Metrics bar chart saved → {path}")
        return path

    def initiate(self, model_path: str, split_path: str) -> dict:
        logger.info("=" * 60)
        logger.info("STAGE 5 ▶ MODEL EVALUATION")
        logger.info("=" * 60)

        try:
            model = load_object(model_path)
            data  = load_object(split_path)
            X_test, y_test = data["X_test"], data["y_test"]

            y_pred = model.predict(X_test)
            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]

            metrics = self._compute_metrics(y_test, y_pred, y_prob)

            logger.info("─── Evaluation Metrics ───────────────────────")
            logger.info(f"  Accuracy  : {metrics['accuracy']:.4f}")
            logger.info(f"  Precision : {metrics['precision']:.4f}")
            logger.info(f"  Recall    : {metrics['recall']:.4f}")
            logger.info(f"  F1-Score  : {metrics['f1_score']:.4f}")
            if metrics.get("roc_auc"):
                logger.info(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
            logger.info("─────────────────────────────────────────────")
            logger.info(f"\n{metrics['classification_report']}")

            # Visualisations
            self._plot_confusion_matrix(y_test, y_pred)
            self._plot_metrics_bar(metrics)

            # Persist report
            report_path = str(Path(self.config.reports_dir) / "evaluation_report.json")
            metrics_to_save = {k: v for k, v in metrics.items() if k != "classification_report"}
            metrics_to_save["classification_report_text"] = metrics["classification_report"]
            save_json(metrics_to_save, report_path)

            logger.info(f"Model Evaluation complete → {report_path}")
            return metrics

        except Exception as e:
            raise ModelEvaluationException(str(e))
