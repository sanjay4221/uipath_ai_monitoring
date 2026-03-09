"""
model_trainer.py
────────────────
Stage 4 Model Training
Trains multiple scikit-learn classifiers on UiPath log features,
selects the best via cross-validation, and saves the winner.
"""

import numpy as np
import joblib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, List

from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from uipath_ai_monitoring.logger import logger
from uipath_ai_monitoring.exception import ModelTrainingException
from uipath_ai_monitoring.utils import save_object, save_json


@dataclass
class ModelTrainerConfig:
    model_dir: str
    reports_dir: str
    test_size: float = 0.20
    random_state: int = 42
    cv_folds: int = 5
    model_names: List[str] = field(default_factory=lambda: [
        "RandomForest", "LogisticRegression", "GradientBoosting"
    ])


class ModelTrainer:
    """Train, cross-validate, and persist the best UiPath error classifier."""

    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        Path(config.model_dir).mkdir(parents=True, exist_ok=True)
        Path(config.reports_dir).mkdir(parents=True, exist_ok=True)

    def _build_candidates(self) -> Dict:
        rs = self.config.random_state
        return {
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                class_weight="balanced",
                random_state=rs,
                n_jobs=-1,
            ),
            "LogisticRegression": LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=rs,
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                random_state=rs,
            ),
        }

    def _cross_validate(self, model, X, y) -> Dict:
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True,
                             random_state=self.config.random_state)
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
        return {
            "mean_cv_f1": float(np.mean(scores)),
            "std_cv_f1":  float(np.std(scores)),
            "cv_scores":  [float(s) for s in scores],
        }

    def initiate(self, X_path: str, y_path: str) -> Tuple[str, str]:
        logger.info("=" * 60)
        logger.info("STAGE 4 MODEL TRAINING")
        logger.info("=" * 60)

        try:
            # Load features
            X = load_npz(X_path)
            y = np.load(y_path)
            logger.info(f"Loaded features: {X.shape}, target: {y.shape}")

            # Train / test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y,
            )
            logger.info(f"Train={X_train.shape[0]}  Test={X_test.shape[0]}")

            candidates = self._build_candidates()
            cv_results: Dict[str, Dict] = {}
            best_name, best_score, best_model = None, -1.0, None

            for name, model in candidates.items():
                logger.info(f"  Cross-validating: {name} ...")
                cv = self._cross_validate(model, X_train, y_train)
                cv_results[name] = cv
                logger.info(f"    CV F1 = {cv['mean_cv_f1']:.4f} ± {cv['std_cv_f1']:.4f}")

                if cv["mean_cv_f1"] > best_score:
                    best_score = cv["mean_cv_f1"]
                    best_name  = name
                    best_model = model

            logger.info(f"Best model: {best_name} (CV F1={best_score:.4f})")

            # Final fit on full training set
            best_model.fit(X_train, y_train)

            # Quick sanity check on test set
            y_pred = best_model.predict(X_test)
            report_str = classification_report(y_test, y_pred,
                                               target_names=["Non-Error", "Error"])
            logger.info(f"Test-set classification report:\n{report_str}")

            # Save model
            model_path = str(Path(self.config.model_dir) / "best_model.pkl")
            save_object(best_model, model_path)

            # Save training artefacts for evaluation stage
            split_path = str(Path(self.config.model_dir) / "train_test_split.pkl")
            save_object(
                {"X_train": X_train, "X_test": X_test,
                 "y_train": y_train, "y_test": y_test},
                split_path,
            )

            # Save CV results
            training_report = {
                "best_model": best_name,
                "best_cv_f1": best_score,
                "cv_results": cv_results,
            }
            save_json(training_report,
                      str(Path(self.config.reports_dir) / "training_report.json"))

            logger.info(f"Model Training complete → {model_path}")
            return model_path, split_path

        except Exception as e:
            raise ModelTrainingException(str(e))
