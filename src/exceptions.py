"""
Custom exceptions for the application.
"""

from typing import Optional
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SentinelAIError(Exception):
    """Base exception for all application errors."""
    def __init__(self, message: str, original: Optional[Exception] = None):
        super().__init__(message)
        self.original = original


class DataLoadError(SentinelAIError):
    """Raised when data loading or processing fails."""
    pass


class ConfigurationError(SentinelAIError):
    """Raised for configuration-related issues."""
    pass


class ModelError(SentinelAIError):
    """Raised during model training or prediction."""
    pass


class UIError(SentinelAIError):
    """Raised for UI state or widget errors."""
    pass


@contextmanager
def error_context(context: str):
    """Context manager to log and wrap exceptions."""
    try:
        yield
    except Exception as e:
        logger.exception(f"Error in {context}: {e}")
        raise DataLoadError(f"{context} failed: {str(e)}") from e
