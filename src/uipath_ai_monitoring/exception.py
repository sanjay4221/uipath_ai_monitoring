"""
exception.py
────────────
Custom exception hierarchy for UiPath AI Monitoring.
"""

import sys


def _error_detail(error: Exception, error_detail: sys) -> str:
    """Extract filename, line number, and message from the traceback."""
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "unknown"
    line_number = exc_tb.tb_lineno if exc_tb else -1
    return (
        f"Error in script: [{file_name}] "
        f"at line [{line_number}] → {str(error)}"
    )


class UiPathAIException(Exception):
    """Base exception for all pipeline errors."""

    def __init__(self, error_message: str, error_detail: sys = sys):
        super().__init__(error_message)
        self.error_message = _error_detail(Exception(error_message), error_detail)

    def __str__(self) -> str:
        return self.error_message


class DataIngestionException(UiPathAIException):
    """Raised when data loading/ingestion fails."""


class DataValidationException(UiPathAIException):
    """Raised when schema or quality checks fail."""


class FeatureEngineeringException(UiPathAIException):
    """Raised during feature transformation."""


class ModelTrainingException(UiPathAIException):
    """Raised during model training."""


class ModelEvaluationException(UiPathAIException):
    """Raised during model evaluation."""


class GroqAIException(UiPathAIException):
    """Raised when Groq API calls fail."""
