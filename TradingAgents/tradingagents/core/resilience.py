"""
Resilience patterns for trading operations.

This module provides retry mechanisms, timeout management, and connection
pooling specifically designed for trading system reliability.
"""

import asyncio
import time
import random
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, Optional, Union, Awaitable, TypeVar, Generic
from dataclasses import dataclass
from functools import wraps
import logging

from .exceptions import TradingSystemError, ErrorSeverity, RecoveryAction

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    max_attempts: int = 3
    base_delay: float = 1.0          # Base delay in seconds
    max_delay: float = 60.0          # Maximum delay in seconds  
    exponential_base: float = 2.0    # Exponential backoff base
    jitter: bool = True              # Add random jitter
    retryable_exceptions: tuple = (ConnectionError, TimeoutError, TradingSystemError)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.base_delay <= 0:
            raise ValueError("base_delay must be positive")
        if self.max_delay <= self.base_delay:
            raise ValueError("max_delay must be greater than base_delay")


class RetryWithBackoff:
    """Retry mechanism with exponential backoff and jitter."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for adding retry behavior to functions."""
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            return await self._retry_async(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            return self._retry_sync(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    async def _retry_async(self, func: Callable, *args, **kwargs) -> T:
        """Async retry implementation."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                if not self._should_retry(e, attempt):
                    logger.error(
                        f"Retry failed permanently after {attempt + 1} attempts",
                        extra={
                            'function': func.__name__,
                            'attempt': attempt + 1,
                            'error': str(e),
                            'error_type': type(e).__name__
                        }
                    )
                    raise
                
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Retry attempt {attempt + 1}/{self.config.max_attempts} "
                    f"failed, retrying in {delay:.2f}s",
                    extra={
                        'function': func.__name__,
                        'attempt': attempt + 1,
                        'delay': delay,
                        'error': str(e)
                    }
                )
                
                await asyncio.sleep(delay)
        
        # All attempts failed
        raise last_exception
    
    def _retry_sync(self, func: Callable, *args, **kwargs) -> T:
        """Synchronous retry implementation."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return func(*args, **kwargs)
                    
            except Exception as e:
                last_exception = e
                
                if not self._should_retry(e, attempt):
                    logger.error(
                        f"Retry failed permanently after {attempt + 1} attempts",
                        extra={
                            'function': func.__name__,
                            'attempt': attempt + 1,
                            'error': str(e),
                            'error_type': type(e).__name__
                        }
                    )
                    raise
                
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Retry attempt {attempt + 1}/{self.config.max_attempts} "
                    f"failed, retrying in {delay:.2f}s",
                    extra={
                        'function': func.__name__,
                        'attempt': attempt + 1,
                        'delay': delay,
                        'error': str(e)
                    }
                )
                
                time.sleep(delay)
        
        # All attempts failed
        raise last_exception
    
    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an exception should trigger a retry."""
        # Don't retry if we've exhausted attempts
        if attempt >= self.config.max_attempts - 1:
            return False
        
        # Check if exception type is retryable
        if not isinstance(exception, self.config.retryable_exceptions):
            return False
        
        # For TradingSystemError, check if it's marked as recoverable
        if isinstance(exception, TradingSystemError):
            return exception.can_retry()
        
        return True
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for next retry with exponential backoff and jitter."""
        # Exponential backoff
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        
        # Cap at max delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)


class TimeoutManager:
    """Context manager for operation timeouts."""
    
    def __init__(self, timeout: float, operation_name: str = "operation"):
        self.timeout = timeout
        self.operation_name = operation_name
        self.start_time = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if exc_type is asyncio.TimeoutError:
            elapsed = time.time() - self.start_time
            logger.error(
                f"Operation '{self.operation_name}' timed out after {elapsed:.2f}s "
                f"(limit: {self.timeout}s)"
            )
            raise TradingSystemError(
                f"Operation '{self.operation_name}' timed out",
                error_code="TIMEOUT",
                severity=ErrorSeverity.MEDIUM,
                recovery_action=RecoveryAction.RETRY,
                context={
                    'operation': self.operation_name,
                    'timeout': self.timeout,
                    'elapsed': elapsed
                }
            )
    
    def __enter__(self):
        """Sync context manager entry."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit."""
        # For sync operations, we can't prevent timeout but can log it
        if self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout:
                logger.warning(
                    f"Operation '{self.operation_name}' exceeded timeout "
                    f"({elapsed:.2f}s > {self.timeout}s)"
                )
    
    @classmethod
    async def run_with_timeout(
        cls, 
        coro: Awaitable[T], 
        timeout: float, 
        operation_name: str = "operation"
    ) -> T:
        """Run a coroutine with timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(f"Operation '{operation_name}' timed out after {timeout}s")
            raise TradingSystemError(
                f"Operation '{operation_name}' timed out",
                error_code="TIMEOUT",
                severity=ErrorSeverity.MEDIUM,
                recovery_action=RecoveryAction.RETRY,
                context={
                    'operation': operation_name,
                    'timeout': timeout
                }
            )


class ConnectionPool(Generic[T]):
    """Generic connection pool for managing broker connections."""
    
    def __init__(
        self,
        create_connection: Callable[[], T],
        max_connections: int = 10,
        min_connections: int = 2,
        max_idle_time: float = 300.0,  # 5 minutes
        connection_timeout: float = 30.0,
        health_check: Optional[Callable[[T], bool]] = None
    ):
        self.create_connection = create_connection
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.max_idle_time = max_idle_time
        self.connection_timeout = connection_timeout
        self.health_check = health_check
        
        self._pool: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._created_connections = 0
        
        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def get_connection(self) -> T:
        """Get a connection from the pool."""
        async with self._lock:
            # Try to get a healthy connection from the pool
            while self._pool:
                conn_info = self._pool.pop(0)
                connection = conn_info['connection']
                
                # Check if connection is still healthy
                if self._is_connection_healthy(connection):
                    logger.debug(
                        "Retrieved connection from pool",
                        extra={'pool_size': len(self._pool)}
                    )
                    return connection
                else:
                    logger.debug("Discarded unhealthy connection from pool")
                    self._created_connections -= 1
            
            # No healthy connections in pool, create new one
            if self._created_connections < self.max_connections:
                try:
                    connection = await asyncio.wait_for(
                        self._create_connection_async(),
                        timeout=self.connection_timeout
                    )
                    self._created_connections += 1
                    logger.debug(
                        "Created new connection",
                        extra={
                            'total_connections': self._created_connections,
                            'max_connections': self.max_connections
                        }
                    )
                    return connection
                except Exception as e:
                    logger.error(f"Failed to create connection: {e}")
                    raise TradingSystemError(
                        "Failed to create database connection",
                        error_code="CONNECTION_CREATION_FAILED",
                        severity=ErrorSeverity.HIGH,
                        recovery_action=RecoveryAction.RETRY,
                        context={'error': str(e)},
                        cause=e
                    )
            else:
                # Pool is at capacity
                raise TradingSystemError(
                    "Connection pool exhausted",
                    error_code="POOL_EXHAUSTED",
                    severity=ErrorSeverity.HIGH,
                    recovery_action=RecoveryAction.RETRY,
                    context={
                        'max_connections': self.max_connections,
                        'created_connections': self._created_connections
                    }
                )
    
    async def return_connection(self, connection: T):
        """Return a connection to the pool."""
        async with self._lock:
            # Check if connection is still healthy
            if self._is_connection_healthy(connection):
                self._pool.append({
                    'connection': connection,
                    'returned_at': datetime.now(timezone.utc)
                })
                logger.debug(
                    "Returned connection to pool",
                    extra={'pool_size': len(self._pool)}
                )
            else:
                logger.debug("Discarded unhealthy connection instead of returning to pool")
                self._created_connections -= 1
    
    async def _create_connection_async(self) -> T:
        """Create connection asynchronously."""
        if asyncio.iscoroutinefunction(self.create_connection):
            return await self.create_connection()
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.create_connection)
    
    def _is_connection_healthy(self, connection: T) -> bool:
        """Check if a connection is healthy."""
        if self.health_check:
            try:
                return self.health_check(connection)
            except Exception as e:
                logger.debug(f"Health check failed: {e}")
                return False
        return True
    
    async def _cleanup_loop(self):
        """Background task to clean up idle connections."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_idle_connections()
            except Exception as e:
                logger.error(f"Error in connection cleanup: {e}")
    
    async def _cleanup_idle_connections(self):
        """Remove idle connections from the pool."""
        async with self._lock:
            now = datetime.now(timezone.utc)
            active_connections = []
            
            for conn_info in self._pool:
                idle_time = (now - conn_info['returned_at']).total_seconds()
                
                if (idle_time < self.max_idle_time and 
                    self._is_connection_healthy(conn_info['connection'])):
                    active_connections.append(conn_info)
                else:
                    logger.debug(
                        "Cleaned up idle connection",
                        extra={'idle_time': idle_time}
                    )
                    self._created_connections -= 1
            
            # Keep minimum number of connections
            while (len(active_connections) < self.min_connections and 
                   self._created_connections < self.max_connections):
                try:
                    connection = await self._create_connection_async()
                    active_connections.append({
                        'connection': connection,
                        'returned_at': now
                    })
                    self._created_connections += 1
                except Exception as e:
                    logger.error(f"Failed to create minimum connection: {e}")
                    break
            
            self._pool = active_connections
            logger.debug(
                "Connection cleanup completed",
                extra={
                    'pool_size': len(self._pool),
                    'total_connections': self._created_connections
                }
            )
    
    async def close(self):
        """Close all connections and cleanup."""
        self._cleanup_task.cancel()
        
        async with self._lock:
            for conn_info in self._pool:
                connection = conn_info['connection']
                # Close connection if it has a close method
                if hasattr(connection, 'close'):
                    try:
                        if asyncio.iscoroutinefunction(connection.close):
                            await connection.close()
                        else:
                            connection.close()
                    except Exception as e:
                        logger.error(f"Error closing connection: {e}")
            
            self._pool.clear()
            self._created_connections = 0
        
        logger.info("Connection pool closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            'pool_size': len(self._pool),
            'total_connections': self._created_connections,
            'max_connections': self.max_connections,
            'min_connections': self.min_connections,
            'utilization': self._created_connections / self.max_connections
        }


# Convenience decorators
def retry_on_failure(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple = (ConnectionError, TimeoutError, TradingSystemError)
):
    """Decorator for retrying trading operations."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        retryable_exceptions=retryable_exceptions
    )
    return RetryWithBackoff(config)


def timeout_operation(timeout: float, operation_name: str = "operation"):
    """Decorator for adding timeout to operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await TimeoutManager.run_with_timeout(
                func(*args, **kwargs), 
                timeout, 
                operation_name
            )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with TimeoutManager(timeout, operation_name):
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator