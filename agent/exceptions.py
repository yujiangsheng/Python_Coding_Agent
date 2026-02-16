"""
exceptions.py — Custom exception hierarchy for PyCoder.

Provides structured error types so callers can catch specific failures
instead of relying on generic Exception.
"""


import logging

class PyCoderError(Exception):
    """Base exception for PyCoder agent."""
    error_code = "PyCoderError"
    
    def __init__(self, message="PyCoder error", cause=None, error_code=None, *args):
        self.message = message
        if error_code is not None:
            if not isinstance(error_code, str):
                raise TypeError("error_code must be a string")
            if not error_code.strip():
                raise ValueError("error_code cannot be empty")
        self.error_code = error_code or self.__class__.error_code
        self.cause = cause
        self.traceback = None
        logging.error(f"{self.error_code}: {message}")
        if cause is not None:
            logging.error(f"Cause: {cause}")
        else:
            logging.error("Cause: None")
        super().__init__(message, *args)

    def get_formatted_error(self):
        formatted = f"{self.error_code}: {self.message}"
        if self.cause:
            formatted += f" (Cause: {self.cause})"
        return formatted

    def get_error_code(self):
        return self.error_code


class ModelNotLoadedError(PyCoderError):
    """Raised when attempting inference before the model is loaded."""
    def __init__(self, message="Model not loaded", context=None, *args):
        self.message = message
        if context is not None and not isinstance(context, dict):
            raise TypeError("Context must be a dictionary or None")
        self.context = context if isinstance(context, dict) else {}
        self.retry_history = []
        super().__init__(message, *args)

    def get_context_info(self):
        return self.context

    def get_context_as_string(self):
        if not self.context:
            return "No context provided"
        return ", ".join([f"{k}: {v}" for k, v in self.context.items()])

    def add_retry_attempt(self, attempt_info):
        self.retry_history.append(attempt_info)


class BackendUnavailableError(PyCoderError):
    """Raised when no usable backend (Ollama / Transformers) is found."""
    def __init__(self, message="Backend unavailable", *args):
        self.message = message
        super().__init__(message, *args)

class RateLimitError(PyCoderError):
    """Raised when API rate limits are exceeded."""
    def __init__(self, message="Rate limit exceeded", retry_after=None, max_retries=3, retry_count=0, *args):
        self.message = message
        self.retry_after = retry_after
        self.max_retries = max_retries
        self.retry_count = retry_count
        self.retry_history = []
        super().__init__(message, *args)

    def get_retry_delay(self):
        if self.retry_after is not None and self.retry_after >= 0:
            return self.retry_after
        import random
        base_delay = 2 ** self.retry_count
        jitter = random.uniform(0, 1)
        delay = min(base_delay + jitter, 60)  # Cap at 60 seconds
        logging.info(f"Rate limit retry delay: {delay} seconds")
        return delay

    def get_next_retry_delay(self):
        """Get the next retry delay without modifying internal state."""
        if self.retry_after is not None and self.retry_after >= 0:
            return self.retry_after
        import random
        base_delay = 2 ** self.retry_count
        jitter = random.random()
        delay = min(base_delay + jitter, 60)  # Cap at 60 seconds
        return delay

    def retry(self):
        import time
        if self.retry_count < self.max_retries:
            delay = self.get_retry_delay()
            time.sleep(delay)
            self.retry_count += 1
            self.retry_history.append(delay)
            logging.info(f"Retrying after {delay} seconds (attempt {self.retry_count}/{self.max_retries})")
            return True
        raise RuntimeError("Max retries exceeded")

    def can_retry(self):
        return self.retry_count < self.max_retries

    def log_retry_attempt(self):
        logging.info(f"Retry attempt {self.retry_count} out of {self.max_retries} for rate limit error")

    def is_exhausted(self):
        return self.retry_count >= self.max_retries

    def get_retry_info(self):
        return {
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "retry_history": self.retry_history,
            "can_retry": self.can_retry()
        }

    def get_full_retry_info(self):
        return {
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "retry_history": self.retry_history,
            "can_retry": self.can_retry(),
            "next_retry_delay": self.get_retry_delay()
        }


import traceback

class GenerationError(PyCoderError):
    """Raised when text/code generation fails."""
    def __init__(self, message="Generation failed", context=None, max_retries=3, retry_count=0, *args):
        self.message = message
        self.context = context or {}
        self.max_retries = max_retries
        self.retry_count = retry_count
        self.retry_history = []
        self.traceback = traceback.format_exc()
        if context:
            message = f"{message}. Context: {context}"
        super().__init__(message, *args)


class ConfigError(PyCoderError):
    """Raised for missing or invalid configuration."""
    def __init__(self, message="Configuration error", *args):
        self.message = message
        super().__init__(message, *args)
