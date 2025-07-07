"""
Trading system exception hierarchy with context and recovery patterns.

This module provides a comprehensive set of exceptions specifically designed
for futures trading operations, with built-in context and recovery guidance.
"""

import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels for trading operations."""
    LOW = "low"           # Retry operation, continue trading
    MEDIUM = "medium"     # Pause strategy, assess situation
    HIGH = "high"         # Stop strategy, manual intervention
    CRITICAL = "critical" # Emergency shutdown, all positions flat


class RecoveryAction(Enum):
    """Recommended recovery actions for different error types."""
    RETRY = "retry"
    RECONNECT = "reconnect"
    PAUSE_STRATEGY = "pause_strategy"
    STOP_STRATEGY = "stop_strategy"
    EMERGENCY_FLATTEN = "emergency_flatten"
    MANUAL_INTERVENTION = "manual_intervention"


class TradingSystemError(Exception):
    """Base exception for all trading system errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recovery_action: RecoveryAction = RecoveryAction.MANUAL_INTERVENTION,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = True,
        retry_count: int = 0,
        max_retries: int = 3
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.severity = severity
        self.recovery_action = recovery_action
        self.context = context or {}
        self.cause = cause
        self.recoverable = recoverable
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.timestamp = datetime.now(timezone.utc)
        
        # Add system context
        self.context.update({
            'timestamp': self.timestamp.isoformat(),
            'error_class': self.__class__.__name__,
            'retry_count': retry_count,
            'max_retries': max_retries,
        })
    
    def can_retry(self) -> bool:
        """Check if this error can be retried."""
        return self.recoverable and self.retry_count < self.max_retries
    
    def get_context_summary(self) -> str:
        """Get a human-readable summary of the error context."""
        summary = [f"Error: {self.message}"]
        summary.append(f"Severity: {self.severity.value}")
        summary.append(f"Recovery: {self.recovery_action.value}")
        
        if self.context:
            summary.append("Context:")
            for key, value in self.context.items():
                if key != 'timestamp':  # Skip timestamp as it's already shown
                    summary.append(f"  {key}: {value}")
        
        return "\n".join(summary)


class MarketDataError(TradingSystemError):
    """Errors related to market data feeds and processing."""
    
    def __init__(
        self,
        message: str,
        symbol: Optional[str] = None,
        data_source: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'symbol': symbol,
            'data_source': data_source,
        })
        kwargs['context'] = context
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('recovery_action', RecoveryAction.RETRY)
        super().__init__(message, **kwargs)


class BrokerConnectionError(TradingSystemError):
    """Errors related to broker connectivity and API issues."""
    
    def __init__(
        self,
        message: str,
        broker: Optional[str] = None,
        connection_id: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'broker': broker,
            'connection_id': connection_id,
        })
        kwargs['context'] = context
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('recovery_action', RecoveryAction.RECONNECT)
        super().__init__(message, **kwargs)


class RiskManagementError(TradingSystemError):
    """Errors related to risk management violations."""
    
    def __init__(
        self,
        message: str,
        risk_type: Optional[str] = None,
        current_value: Optional[float] = None,
        limit_value: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'risk_type': risk_type,
            'current_value': current_value,
            'limit_value': limit_value,
        })
        kwargs['context'] = context
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('recovery_action', RecoveryAction.STOP_STRATEGY)
        kwargs.setdefault('recoverable', False)  # Risk violations usually not retryable
        super().__init__(message, **kwargs)


class StrategyError(TradingSystemError):
    """Errors related to trading strategy execution."""
    
    def __init__(
        self,
        message: str,
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'strategy_name': strategy_name,
            'symbol': symbol,
        })
        kwargs['context'] = context
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('recovery_action', RecoveryAction.PAUSE_STRATEGY)
        super().__init__(message, **kwargs)


class OrderExecutionError(TradingSystemError):
    """Errors related to order placement and execution."""
    
    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        order_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'order_id': order_id,
            'symbol': symbol,
            'order_type': order_type,
        })
        kwargs['context'] = context
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('recovery_action', RecoveryAction.RETRY)
        super().__init__(message, **kwargs)


class ConfigurationError(TradingSystemError):
    """Errors related to system configuration and setup."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'config_key': config_key,
            'config_value': str(config_value) if config_value is not None else None,
        })
        kwargs['context'] = context
        kwargs.setdefault('severity', ErrorSeverity.CRITICAL)
        kwargs.setdefault('recovery_action', RecoveryAction.MANUAL_INTERVENTION)
        kwargs.setdefault('recoverable', False)
        super().__init__(message, **kwargs)


class PositionManagementError(TradingSystemError):
    """Errors related to position management and tracking."""
    
    def __init__(
        self,
        message: str,
        symbol: Optional[str] = None,
        position_size: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'symbol': symbol,
            'position_size': position_size,
        })
        kwargs['context'] = context
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('recovery_action', RecoveryAction.PAUSE_STRATEGY)
        super().__init__(message, **kwargs)


class ValidationError(TradingSystemError):
    """Errors related to data and parameter validation."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'field_name': field_name,
            'field_value': str(field_value) if field_value is not None else None,
        })
        kwargs['context'] = context
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('recovery_action', RecoveryAction.MANUAL_INTERVENTION)
        super().__init__(message, **kwargs)


# Emergency situations
class EmergencyShutdownError(TradingSystemError):
    """Critical errors requiring immediate system shutdown."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('severity', ErrorSeverity.CRITICAL)
        kwargs.setdefault('recovery_action', RecoveryAction.EMERGENCY_FLATTEN)
        kwargs.setdefault('recoverable', False)
        super().__init__(message, **kwargs)


# Utility functions for error handling
def create_error_context(
    symbol: Optional[str] = None,
    strategy: Optional[str] = None,
    additional_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a standard error context dictionary."""
    context = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'symbol': symbol,
        'strategy': strategy,
    }
    
    if additional_context:
        context.update(additional_context)
    
    # Remove None values
    return {k: v for k, v in context.items() if v is not None}


def handle_trading_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    default_recovery: RecoveryAction = RecoveryAction.MANUAL_INTERVENTION
) -> TradingSystemError:
    """Convert any exception to a TradingSystemError with context."""
    
    if isinstance(error, TradingSystemError):
        return error
    
    # Convert common exceptions to appropriate trading errors
    if isinstance(error, ConnectionError):
        return BrokerConnectionError(
            f"Connection error: {str(error)}",
            context=context,
            cause=error
        )
    elif isinstance(error, ValueError):
        return ValidationError(
            f"Validation error: {str(error)}",
            context=context,
            cause=error
        )
    elif isinstance(error, TimeoutError):
        return MarketDataError(
            f"Timeout error: {str(error)}",
            context=context,
            cause=error
        )
    else:
        # Generic trading system error
        return TradingSystemError(
            f"Unexpected error: {str(error)}",
            context=context,
            cause=error,
            recovery_action=default_recovery
        )