"""
Custom exceptions for TradingAgents.

This module defines application-specific exceptions with proper error hierarchy
and context information for better error handling and debugging.
"""

from typing import Any, Dict, Optional
import traceback


class TradingAgentsError(Exception):
    """Base exception class for all TradingAgents errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        """Initialize the base exception.
        
        Args:
            message: Human-readable error message
            error_code: Unique error code for categorization
            context: Additional context information
            original_exception: Original exception if this is a wrapped error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.original_exception = original_exception
        
        if original_exception:
            self.context['original_traceback'] = traceback.format_exception(
                type(original_exception), original_exception, original_exception.__traceback__
            )


class ConfigurationError(TradingAgentsError):
    """Raised when there's a configuration-related error."""
    pass


class APIError(TradingAgentsError):
    """Base class for API-related errors."""
    pass


class APIConnectionError(APIError):
    """Raised when unable to connect to an external API."""
    pass


class APIRateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    
    def __init__(
        self, 
        message: str, 
        retry_after: Optional[int] = None,
        **kwargs
    ):
        """Initialize rate limit error.
        
        Args:
            message: Error message
            retry_after: Seconds to wait before retrying
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class APIResponseError(APIError):
    """Raised when API returns an unexpected response."""
    
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs
    ):
        """Initialize API response error.
        
        Args:
            message: Error message
            status_code: HTTP status code
            response_body: Response body content
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.response_body = response_body


class DataError(TradingAgentsError):
    """Base class for data-related errors."""
    pass


class DataValidationError(DataError):
    """Raised when data validation fails."""
    pass


class DataNotFoundError(DataError):
    """Raised when required data is not found."""
    pass


class DataCorruptionError(DataError):
    """Raised when data appears to be corrupted."""
    pass


class AgentError(TradingAgentsError):
    """Base class for agent-related errors."""
    pass


class AgentTimeoutError(AgentError):
    """Raised when an agent operation times out."""
    pass


class AgentCommunicationError(AgentError):
    """Raised when agents fail to communicate properly."""
    pass


class TradingError(TradingAgentsError):
    """Base class for trading-related errors."""
    pass


class InsufficientFundsError(TradingError):
    """Raised when there are insufficient funds for a trade."""
    pass


class InvalidTradeError(TradingError):
    """Raised when a trade request is invalid."""
    pass


class RiskLimitExceededError(TradingError):
    """Raised when a trade would exceed risk limits."""
    pass


class SecurityError(TradingAgentsError):
    """Base class for security-related errors."""
    pass


class AuthenticationError(SecurityError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(SecurityError):
    """Raised when authorization fails."""
    pass


class ModelError(TradingAgentsError):
    """Base class for LLM model-related errors."""
    pass


class ModelTimeoutError(ModelError):
    """Raised when model response times out."""
    pass


class ModelOverloadError(ModelError):
    """Raised when model is overloaded."""
    pass


class TokenLimitError(ModelError):
    """Raised when token limit is exceeded."""
    pass