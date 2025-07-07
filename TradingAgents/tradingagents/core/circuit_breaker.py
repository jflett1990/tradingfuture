"""
Circuit breaker pattern implementation for trading operations.

This module provides circuit breakers specifically designed for trading
systems with customizable failure thresholds and recovery strategies.
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union, Awaitable
from dataclasses import dataclass, field
import structlog

from .exceptions import TradingSystemError, ErrorSeverity

logger = structlog.get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"     # Normal operation
    OPEN = "open"         # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    
    # Failure thresholds
    failure_threshold: int = 5           # Failures before opening
    recovery_timeout: float = 60.0       # Seconds before trying half-open
    success_threshold: int = 3           # Successes to close from half-open
    
    # Timeouts
    request_timeout: float = 30.0        # Individual request timeout
    
    # Monitoring
    window_size: int = 100              # Rolling window for failure rate
    failure_rate_threshold: float = 0.5  # Rate that triggers opening
    
    # Trading-specific settings
    emergency_flatten: bool = True       # Flatten positions on critical failures
    pause_trading: bool = True          # Pause new trades when open
    
    def __post_init__(self):
        """Validate configuration."""
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if self.recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be positive")
        if self.success_threshold <= 0:
            raise ValueError("success_threshold must be positive")
        if not 0 < self.failure_rate_threshold <= 1:
            raise ValueError("failure_rate_threshold must be between 0 and 1")


@dataclass
class CircuitBreakerMetrics:
    """Metrics tracked by circuit breaker."""
    
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_failure_time: Optional[datetime] = None
    state_changes: int = 0
    current_state: CircuitState = CircuitState.CLOSED
    
    # Rolling window for failure rate calculation
    recent_results: list = field(default_factory=list)
    
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if not self.recent_results:
            return 0.0
        failures = sum(1 for result in self.recent_results if not result)
        return failures / len(self.recent_results)
    
    def add_result(self, success: bool, window_size: int = 100):
        """Add a request result to the rolling window."""
        self.recent_results.append(success)
        if len(self.recent_results) > window_size:
            self.recent_results.pop(0)
        
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            self.last_failure_time = datetime.now(timezone.utc)


class CircuitBreaker:
    """
    Circuit breaker implementation for trading operations.
    
    Provides protection against cascading failures and allows for
    graceful degradation in trading systems.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback_function: Optional[Callable] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback_function = fallback_function
        
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_state_change = datetime.now(timezone.utc)
        
        self._lock = asyncio.Lock()
        
        logger.info(
            "circuit_breaker_created",
            name=self.name,
            config=self.config.__dict__
        )
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
            
        Returns:
            Function result or fallback result
            
        Raises:
            TradingSystemError: When circuit is open and no fallback
        """
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    await self._transition_to_half_open()
                else:
                    return await self._handle_open_circuit(func, *args, **kwargs)
            
            # Execute the function
            try:
                # Set timeout for trading operations
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.config.request_timeout
                    )
                else:
                    result = func(*args, **kwargs)
                
                await self._record_success()
                return result
                
            except Exception as e:
                await self._record_failure(e)
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        time_since_last_change = (
            datetime.now(timezone.utc) - self.last_state_change
        ).total_seconds()
        return time_since_last_change >= self.config.recovery_timeout
    
    async def _transition_to_half_open(self):
        """Transition circuit to half-open state."""
        self.state = CircuitState.HALF_OPEN
        self.consecutive_successes = 0
        self.last_state_change = datetime.now(timezone.utc)
        self.metrics.state_changes += 1
        
        logger.info(
            "circuit_breaker_half_open",
            name=self.name,
            failure_count=self.consecutive_failures
        )
    
    async def _handle_open_circuit(self, func: Callable, *args, **kwargs) -> Any:
        """Handle requests when circuit is open."""
        logger.warning(
            "circuit_breaker_blocked",
            name=self.name,
            function=func.__name__ if hasattr(func, '__name__') else str(func)
        )
        
        if self.fallback_function:
            try:
                if asyncio.iscoroutinefunction(self.fallback_function):
                    return await self.fallback_function(*args, **kwargs)
                else:
                    return self.fallback_function(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "circuit_breaker_fallback_failed",
                    name=self.name,
                    error=str(e)
                )
                raise
        
        # No fallback available
        raise TradingSystemError(
            f"Circuit breaker '{self.name}' is open - service unavailable",
            error_code="CIRCUIT_OPEN",
            severity=ErrorSeverity.HIGH,
            context={
                'circuit_name': self.name,
                'consecutive_failures': self.consecutive_failures,
                'failure_rate': self.metrics.failure_rate(),
            }
        )
    
    async def _record_success(self):
        """Record a successful operation."""
        self.consecutive_failures = 0
        self.consecutive_successes += 1
        self.metrics.add_result(True, self.config.window_size)
        
        # Transition from half-open to closed if enough successes
        if (self.state == CircuitState.HALF_OPEN and 
            self.consecutive_successes >= self.config.success_threshold):
            await self._transition_to_closed()
    
    async def _record_failure(self, exception: Exception):
        """Record a failed operation."""
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        self.metrics.add_result(False, self.config.window_size)
        
        logger.warning(
            "circuit_breaker_failure",
            name=self.name,
            consecutive_failures=self.consecutive_failures,
            failure_rate=self.metrics.failure_rate(),
            error=str(exception)
        )
        
        # Check if we should open the circuit
        should_open = (
            self.consecutive_failures >= self.config.failure_threshold or
            self.metrics.failure_rate() >= self.config.failure_rate_threshold
        )
        
        if should_open and self.state != CircuitState.OPEN:
            await self._transition_to_open(exception)
    
    async def _transition_to_closed(self):
        """Transition circuit to closed state."""
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.last_state_change = datetime.now(timezone.utc)
        self.metrics.state_changes += 1
        
        logger.info(
            "circuit_breaker_closed",
            name=self.name,
            success_count=self.consecutive_successes
        )
    
    async def _transition_to_open(self, exception: Exception):
        """Transition circuit to open state."""
        self.state = CircuitState.OPEN
        self.last_state_change = datetime.now(timezone.utc)
        self.metrics.state_changes += 1
        
        logger.error(
            "circuit_breaker_opened",
            name=self.name,
            consecutive_failures=self.consecutive_failures,
            failure_rate=self.metrics.failure_rate(),
            trigger_error=str(exception)
        )
        
        # Trading-specific actions
        if isinstance(exception, TradingSystemError):
            if (exception.severity == ErrorSeverity.CRITICAL and 
                self.config.emergency_flatten):
                logger.critical(
                    "circuit_breaker_emergency_flatten",
                    name=self.name,
                    error=str(exception)
                )
                # Note: Actual position flattening would be implemented
                # by the trading system that uses this circuit breaker
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics."""
        return {
            'name': self.name,
            'state': self.state.value,
            'total_requests': self.metrics.total_requests,
            'successful_requests': self.metrics.successful_requests,
            'failed_requests': self.metrics.failed_requests,
            'failure_rate': self.metrics.failure_rate(),
            'consecutive_failures': self.consecutive_failures,
            'consecutive_successes': self.consecutive_successes,
            'state_changes': self.metrics.state_changes,
            'last_failure': (
                self.metrics.last_failure_time.isoformat() 
                if self.metrics.last_failure_time else None
            ),
            'last_state_change': self.last_state_change.isoformat(),
            'config': self.config.__dict__
        }
    
    async def reset(self):
        """Manually reset circuit breaker to closed state."""
        async with self._lock:
            self.state = CircuitState.CLOSED
            self.consecutive_failures = 0
            self.consecutive_successes = 0
            self.last_state_change = datetime.now(timezone.utc)
            self.metrics.state_changes += 1
            
            logger.info(
                "circuit_breaker_manual_reset",
                name=self.name
            )
    
    async def force_open(self, reason: str = "Manual override"):
        """Manually force circuit breaker to open state."""
        async with self._lock:
            self.state = CircuitState.OPEN
            self.last_state_change = datetime.now(timezone.utc)
            self.metrics.state_changes += 1
            
            logger.warning(
                "circuit_breaker_forced_open",
                name=self.name,
                reason=reason
            )


# Specialized circuit breakers for trading operations
class MarketDataCircuitBreaker(CircuitBreaker):
    """Circuit breaker specifically for market data operations."""
    
    def __init__(self, name: str = "market_data", **kwargs):
        config = CircuitBreakerConfig(
            failure_threshold=3,      # Quick to open for data issues
            recovery_timeout=30.0,    # Fast recovery attempt
            success_threshold=2,      # Quick to close
            request_timeout=10.0,     # Short timeout for data requests
            emergency_flatten=False,  # Don't flatten on data issues
            pause_trading=True,       # But do pause new trades
            **kwargs
        )
        super().__init__(name, config)


class BrokerConnectionCircuitBreaker(CircuitBreaker):
    """Circuit breaker specifically for broker connections."""
    
    def __init__(self, name: str = "broker_connection", **kwargs):
        config = CircuitBreakerConfig(
            failure_threshold=2,      # Very sensitive to connection issues
            recovery_timeout=60.0,    # Longer recovery time
            success_threshold=3,      # Need several successes to trust
            request_timeout=30.0,     # Longer timeout for orders
            emergency_flatten=True,   # Flatten on connection loss
            pause_trading=True,       # Definitely pause trading
            **kwargs
        )
        super().__init__(name, config)


class StrategyCircuitBreaker(CircuitBreaker):
    """Circuit breaker specifically for trading strategies."""
    
    def __init__(self, name: str, **kwargs):
        config = CircuitBreakerConfig(
            failure_threshold=5,      # More tolerant of strategy failures
            recovery_timeout=300.0,   # Longer cooldown period
            success_threshold=3,      # Standard recovery
            request_timeout=60.0,     # Longer timeout for strategy execution
            emergency_flatten=False,  # Strategy failures don't flatten
            pause_trading=True,       # But do pause the strategy
            **kwargs
        )
        super().__init__(name, config)