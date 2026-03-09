"""
training_pipeline.py
────────────────────
Orchestrates all five ML pipeline stages end-to-end:
  1. Data Ingestion
  2. Data Validation
  3. Feature Engineering
  4. Model Training
  5. Model Evaluation
"""

from uipath_ai_monitoring.logger import logger
from uipath_ai_monitoring.exception import UiPathAIException
from uipath_ai_monitoring.utils import load_config

from uipath_ai_monitoring.components.data_ingestion import (
    DataIngestion, DataIngestionConfig,
)
from uipath_ai_monitoring.components.data_validation import (
    DataValidation, DataValidationConfig,
)
from uipath_ai_monitoring.components.feature_engineering import (
    FeatureEngineering, FeatureEngineeringConfig,
)
from uipath_ai_monitoring.components.model_trainer import (
    ModelTrainer, ModelTrainerConfig,
)
from uipath_ai_monitoring.components.model_evaluation import (
    ModelEvaluation, ModelEvaluationConfig,
)


class TrainingPipeline:
    """Run the complete ML training pipeline."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.cfg = load_config(config_path)

    def run(self):
        logger.info("UiPath AI Monitoring Training Pipeline")

        try:
            # ── Stage 1: Data Ingestion
            ingestion_cfg = DataIngestionConfig(
                raw_data_dir       = self.cfg["paths"]["raw_data_dir"],
                processed_data_dir = self.cfg["paths"]["processed_data_dir"],
                required_columns   = self.cfg["data_ingestion"]["required_columns"],
            )
            raw_path = DataIngestion(ingestion_cfg).initiate()

            # ── Stage 2: Data Validation 
            validation_cfg = DataValidationConfig(
                required_columns = self.cfg["data_ingestion"]["required_columns"],
                valid_log_levels = self.cfg["data_validation"]["valid_log_levels"],
                max_missing_pct  = self.cfg["data_validation"]["max_missing_pct"],
                reports_dir      = self.cfg["paths"]["reports_dir"],
            )
            validated_path, val_ok = DataValidation(validation_cfg).initiate(raw_path)
            if not val_ok:
                logger.warning("Validation completed with issues – proceeding anyway")

            # ── Stage 3: Feature Engineering 
            fe_cfg_raw = self.cfg["feature_engineering"]
            fe_cfg = FeatureEngineeringConfig(
                tfidf_max_features = fe_cfg_raw["tfidf_max_features"],
                ngram_range        = tuple(fe_cfg_raw["tfidf_ngram_range"]),
                min_df             = fe_cfg_raw["min_df"],
                target_column      = fe_cfg_raw["target_column"],
                processed_data_dir = self.cfg["paths"]["processed_data_dir"],
                model_dir          = self.cfg["paths"]["model_dir"],
            )
            X_path, y_path = FeatureEngineering(fe_cfg).initiate(validated_path)

            # ── Stage 4: Model Training 
            mt_cfg_raw = self.cfg["model_training"]
            mt_cfg = ModelTrainerConfig(
                model_dir    = self.cfg["paths"]["model_dir"],
                reports_dir  = self.cfg["paths"]["reports_dir"],
                test_size    = mt_cfg_raw["test_size"],
                random_state = mt_cfg_raw["random_state"],
                cv_folds     = mt_cfg_raw["cv_folds"],
                model_names  = mt_cfg_raw["models"],
            )
            model_path, split_path = ModelTrainer(mt_cfg).initiate(X_path, y_path)

            # ── Stage 5: Model Evaluation 
            eval_cfg = ModelEvaluationConfig(
                reports_dir = self.cfg["paths"]["reports_dir"],
                model_dir   = self.cfg["paths"]["model_dir"],
            )
            metrics = ModelEvaluation(eval_cfg).initiate(model_path, split_path)

            logger.info("Pipeline completed successfully!")
            logger.info(f"Accuracy  : {metrics['accuracy']:.4f}")
            logger.info(f"F1-Score  : {metrics['f1_score']:.4f}")

            return metrics

        except UiPathAIException as e:
            logger.error(f"Pipeline failed: {e}")
            raise
