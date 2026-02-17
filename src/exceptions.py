"""
SentinelAI Custom Exceptions and Error Handling Utilities.
Provides a consistent way to manage and display errors.
"""

import logging
import functools
import traceback
from typing import Optional, Callable, Any, Tuple, Type

# Configure module logger
logger = logging.getLogger(__name__)


# -------------------- Custom Exceptions --------------------

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


# -------------------- Logging Helpers --------------------

def log_error(error: Exception, context: Optional[str] = None) -> None:
    """
    Log an error with full traceback and optional context.
    """
    if context:
        logger.error(f"{context} – {error}")
    else:
        logger.error(error)
    logger.debug("".join(traceback.format_tb(error.__traceback__)))


def friendly_error_message(error: Exception) -> str:
    """
    Convert an exception to a user-friendly message.
    Override this per exception type if needed.
    """
    if isinstance(error, DataLoadError):
        return f"📁 Data error: {error}"
    elif isinstance(error, ConfigurationError):
        return f"⚙️ Configuration error: {error}"
    elif isinstance(error, ModelError):
        return f"🤖 Model error: {error}"
    else:
        return f"❌ Unexpected error: {error}"


# -------------------- Decorator for Streamlit Apps --------------------

def handle_streamlit_errors(
    default_return: Any = None,
    friendly_message: Optional[str] = None,
    exception_types: Tuple[Type[Exception], ...] = (Exception,),
    use_st_error: bool = True,
):
    """
    Decorator to handle exceptions in Streamlit functions.
    Logs the error and optionally shows a Streamlit error message.

    Args:
        default_return: Value to return if an exception occurs.
        friendly_message: Custom message to show in Streamlit.
        exception_types: Tuple of exception classes to catch.
        use_st_error: Whether to call st.error (must be in Streamlit context).
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                log_error(e, context=f"Error in {func.__name__}")
                if use_st_error:
                    import streamlit as st
                    msg = friendly_message or friendly_error_message(e)
                    st.error(msg)
                return default_return
        return wrapper
    return decorator


# -------------------- Context Manager --------------------

class error_context:
    """
    Context manager for error handling with logging and optional UI feedback.
    Example:
        with error_context("loading data", st.error):
            risky_operation()
    """
    def __init__(self, context: str, ui_func: Optional[Callable] = None):
        self.context = context
        self.ui_func = ui_func

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            log_error(exc_val, context=self.context)
            if self.ui_func:
                self.ui_func(f"Error in {self.context}: {exc_val}")
            # Do not suppress – propagate exception
            return False
