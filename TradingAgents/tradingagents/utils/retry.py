"""
Retry mechanisms for TradingAgents.

This module provides:
- Exponential backoff retry logic
- Circuit breaker pattern
- Rate limiting
- Context-aware error handling
"""

import asyncio
import time
import random
from typing import Any, Callable, Optional, Type, Union, List
from functools import wraps
from dataclasses import dataclass
from enum import Enum
import logging
from threading import Lock
from collections import defaultdict, deque

from .exceptions import (
    APIError, APIConnectionError, APIRateLimitError, APIResponseError,
    ModelError, ModelTimeoutError, ModelOverloadError
)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failures exceeded threshold, blocking calls
    HALF_OPEN = "half_open" # Testing if service has recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_multiplier: float = 1.0
    
    # Exceptions that should trigger retries
    retryable_exceptions: tuple = (
        APIConnectionError,
        APIRateLimitError,
        ModelTimeoutError,
        ModelOverloadError
    )
    
    # Exceptions that should not be retried
    non_retryable_exceptions: tuple = (
        APIResponseError,  # Don't retry on authentication errors, etc.
    )


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5      # Number of failures before opening
    recovery_timeout: float = 60.0  # Seconds to wait before trying again
    expected_exception: Type[Exception] = APIError
    

class RateLimiter:
    """Rate limiter using sliding window algorithm."""
    
    def __init__(self, max_calls: int, window_seconds: float = 60.0):
        """Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the window
            window_seconds: Time window in seconds
        """
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.calls = deque()
        self.lock = Lock()
    
    def acquire(self) -> bool:
        """Try to acquire a permit for making a call.
        
        Returns:
            True if call is allowed, False if rate limited
        """
        with self.lock:
            now = time.time()
            
            # Remove old calls outside the window
            while self.calls and self.calls[0] < now - self.window_seconds:
                self.calls.popleft()
            
            # Check if we can make the call
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            
            return False
    
    def wait_time(self) -> float:
        """Calculate how long to wait before the next call is allowed.
        
        Returns:
            Wait time in seconds
        """
        with self.lock:
            if len(self.calls) < self.max_calls:
                return 0.0
            
            # Calculate when the oldest call will expire
            oldest_call = self.calls[0]
            wait_time = (oldest_call + self.window_seconds) - time.time()
            return max(0.0, wait_time)


class CircuitBreaker:
    """Circuit breaker for failing services."""
    
    def __init__(self, config: CircuitBreakerConfig):
        """Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
        """
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.lock = Lock()
        self.logger = logging.getLogger(__name__)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.logger.info("Circuit breaker moving to HALF_OPEN state")
                else:
                    raise APIConnectionError(
                        f"Circuit breaker is OPEN. Service unavailable.",
                        context={"failure_count": self.failure_count}
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.logger.info("Circuit breaker CLOSED - service recovered")
    
    def _on_failure(self):
        """Handle failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.config.failure_threshold:
                if self.state != CircuitState.OPEN:
                    self.state = CircuitState.OPEN
                    self.logger.warning(
                        f"Circuit breaker OPENED after {self.failure_count} failures"
                    )


class RetryManager:
    """Manages retry logic with circuit breakers and rate limiting."""
    
    def __init__(self):
        """Initialize retry manager."""
        self.circuit_breakers = {}
        self.rate_limiters = {}
        self.lock = Lock()
        self.logger = logging.getLogger(__name__)
    
    def get_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Get or create a circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Circuit breaker configuration
            
        Returns:
            Circuit breaker instance
        """
        with self.lock:
            if name not in self.circuit_breakers:
                self.circuit_breakers[name] = CircuitBreaker(config)
            return self.circuit_breakers[name]
    
    def get_rate_limiter(self, name: str, max_calls: int, window_seconds: float = 60.0) -> RateLimiter:
        """Get or create a rate limiter.
        
        Args:
            name: Rate limiter name
            max_calls: Maximum calls per window
            window_seconds: Window size in seconds
            
        Returns:
            Rate limiter instance
        """
        with self.lock:
            key = f"{name}_{max_calls}_{window_seconds}"
            if key not in self.rate_limiters:
                self.rate_limiters[key] = RateLimiter(max_calls, window_seconds)
            return self.rate_limiters[key]


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    rate_limit_calls: Optional[int] = None,
    rate_limit_window: float = 60.0,
    service_name: Optional[str] = None
):
    """Decorator for retry logic with exponential backoff.
    
    Args:
        config: Retry configuration
        circuit_breaker_config: Circuit breaker configuration
        rate_limit_calls: Maximum calls per window for rate limiting
        rate_limit_window: Rate limit window in seconds
        service_name: Name of the service for circuit breaker/rate limiter
        
    Returns:
        Decorated function
    """
    if config is None:
        config = RetryConfig()
    
    retry_manager = _get_retry_manager()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            func_name = service_name or f"{func.__module__}.{func.__name__}"
            logger = logging.getLogger(__name__)
            
            # Set up circuit breaker if configured
            circuit_breaker = None
            if circuit_breaker_config:
                circuit_breaker = retry_manager.get_circuit_breaker(
                    func_name, circuit_breaker_config
                )
            
            # Set up rate limiter if configured
            rate_limiter = None
            if rate_limit_calls:
                rate_limiter = retry_manager.get_rate_limiter(
                    func_name, rate_limit_calls, rate_limit_window
                )
            
            def execute_call():
                # Check rate limit
                if rate_limiter and not rate_limiter.acquire():
                    wait_time = rate_limiter.wait_time()
                    raise APIRateLimitError(
                        f"Rate limit exceeded for {func_name}",
                        retry_after=int(wait_time) + 1
                    )
                
                # Execute with circuit breaker if configured
                if circuit_breaker:
                    return circuit_breaker.call(func, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
            
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    result = execute_call()
                    if attempt > 0:
                        logger.info(f"Call succeeded on attempt {attempt + 1}")
                    return result
                    
                except config.non_retryable_exceptions as e:
                    logger.error(f"Non-retryable error in {func_name}: {e}")
                    raise
                    
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        logger.error(f"All retry attempts failed for {func_name}")
                        break
                    
                    # Calculate delay
                    delay = _calculate_delay(attempt, config)
                    
                    # Special handling for rate limit errors
                    if isinstance(e, APIRateLimitError) and hasattr(e, 'retry_after') and e.retry_after is not None:
                        delay = max(delay, float(e.retry_after))
                    
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func_name}: {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    
                    time.sleep(delay)
                    
                except Exception as e:
                    logger.error(f"Unexpected error in {func_name}: {e}")
                    raise
            
            # All attempts failed
            if last_exception:
                raise last_exception
            else:
                raise APIError(f"All retry attempts failed for {func_name}")
        
        return wrapper
    return decorator


async def async_retry_with_backoff(
    config: Optional[RetryConfig] = None,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    rate_limit_calls: Optional[int] = None,
    rate_limit_window: float = 60.0,
    service_name: Optional[str] = None
):
    """Async version of retry decorator.
    
    Args:
        config: Retry configuration
        circuit_breaker_config: Circuit breaker configuration  
        rate_limit_calls: Maximum calls per window for rate limiting
        rate_limit_window: Rate limit window in seconds
        service_name: Name of the service for circuit breaker/rate limiter
        
    Returns:
        Decorated async function
    """
    if config is None:
        config = RetryConfig()
    
    retry_manager = _get_retry_manager()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            func_name = service_name or f"{func.__module__}.{func.__name__}"
            logger = logging.getLogger(__name__)
            
            # Set up circuit breaker if configured
            circuit_breaker = None
            if circuit_breaker_config:
                circuit_breaker = retry_manager.get_circuit_breaker(
                    func_name, circuit_breaker_config
                )
            
            # Set up rate limiter if configured
            rate_limiter = None
            if rate_limit_calls:
                rate_limiter = retry_manager.get_rate_limiter(
                    func_name, rate_limit_calls, rate_limit_window
                )
            
            async def execute_call():
                # Check rate limit
                if rate_limiter and not rate_limiter.acquire():
                    wait_time = rate_limiter.wait_time()
                    raise APIRateLimitError(
                        f"Rate limit exceeded for {func_name}",
                        retry_after=int(wait_time) + 1
                    )
                
                # Execute with circuit breaker if configured
                if circuit_breaker:
                    # Note: Circuit breaker call is synchronous
                    return circuit_breaker.call(lambda: asyncio.run(func(*args, **kwargs)))
                else:
                    return await func(*args, **kwargs)
            
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    result = await execute_call()
                    if attempt > 0:
                        logger.info(f"Async call succeeded on attempt {attempt + 1}")
                    return result
                    
                except config.non_retryable_exceptions as e:
                    logger.error(f"Non-retryable error in {func_name}: {e}")
                    raise
                    
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        logger.error(f"All async retry attempts failed for {func_name}")
                        break
                    
                    # Calculate delay
                    delay = _calculate_delay(attempt, config)
                    
                    # Special handling for rate limit errors
                    if isinstance(e, APIRateLimitError) and hasattr(e, 'retry_after') and e.retry_after is not None:
                        delay = max(delay, float(e.retry_after))
                    
                    logger.warning(
                        f"Async attempt {attempt + 1} failed for {func_name}: {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    
                    await asyncio.sleep(delay)
                    
                except Exception as e:
                    logger.error(f"Unexpected async error in {func_name}: {e}")
                    raise
            
            # All attempts failed
            if last_exception:
                raise last_exception
            else:
                raise APIError(f"All async retry attempts failed for {func_name}")
        
        return wrapper
    return decorator


def _calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay for exponential backoff.
    
    Args:
        attempt: Attempt number (0-based)
        config: Retry configuration
        
    Returns:
        Delay in seconds
    """
    # Base exponential backoff
    delay = config.base_delay * (config.exponential_base ** attempt)
    delay *= config.backoff_multiplier
    
    # Apply jitter to avoid thundering herd
    if config.jitter:
        jitter = random.uniform(0.5, 1.5)
        delay *= jitter
    
    # Cap at maximum delay
    delay = min(delay, config.max_delay)
    
    return delay


# Global retry manager instance
_retry_manager = None

def _get_retry_manager() -> RetryManager:
    """Get or create the global retry manager."""
    global _retry_manager
    if _retry_manager is None:
        _retry_manager = RetryManager()
    return _retry_manager