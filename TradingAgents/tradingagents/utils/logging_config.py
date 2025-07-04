"""
Logging configuration for TradingAgents.

This module provides structured logging with:
- Multiple log levels and handlers
- Context tracking for agents and operations
- JSON formatting for structured logs
- Performance monitoring
- Security audit logging
"""

import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict


@dataclass
class LogContext:
    """Context information for log entries."""
    agent_id: Optional[str] = None
    operation: Optional[str] = None
    ticker: Optional[str] = None
    trade_date: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class ContextFilter(logging.Filter):
    """Filter to add context information to log records."""
    
    def __init__(self):
        super().__init__()
        self.local = threading.local()
    
    def filter(self, record):
        context = getattr(self.local, 'context', LogContext())
        
        # Add context fields to the record using setattr to avoid type issues
        setattr(record, 'agent_id', context.agent_id)
        setattr(record, 'operation', context.operation)
        setattr(record, 'ticker', context.ticker)
        setattr(record, 'trade_date', context.trade_date)
        setattr(record, 'session_id', context.session_id)
        setattr(record, 'user_id', context.user_id)
        
        return True
    
    def set_context(self, context: LogContext):
        """Set context for current thread."""
        self.local.context = context
    
    def clear_context(self):
        """Clear context for current thread."""
        self.local.context = LogContext()


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add context fields if available
        if hasattr(record, 'agent_id') and record.agent_id:
            log_entry['agent_id'] = record.agent_id
        if hasattr(record, 'operation') and record.operation:
            log_entry['operation'] = record.operation
        if hasattr(record, 'ticker') and record.ticker:
            log_entry['ticker'] = record.ticker
        if hasattr(record, 'trade_date') and record.trade_date:
            log_entry['trade_date'] = record.trade_date
        if hasattr(record, 'session_id') and record.session_id:
            log_entry['session_id'] = record.session_id
        if hasattr(record, 'user_id') and record.user_id:
            log_entry['user_id'] = record.user_id
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'exc_info', 'exc_text',
                          'stack_info', 'agent_id', 'operation', 'ticker',
                          'trade_date', 'session_id', 'user_id']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class TradingAgentsLogger:
    """Main logger class for TradingAgents."""
    
    def __init__(self, logs_dir: Optional[Path] = None):
        """Initialize the logger.
        
        Args:
            logs_dir: Directory to store log files
        """
        self.logs_dir = logs_dir or Path("./logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.context_filter = ContextFilter()
        self._setup_loggers()
    
    def _setup_loggers(self):
        """Set up different loggers for different purposes."""
        
        # Main application logger
        self.app_logger = self._create_logger(
            'tradingagents.app',
            self.logs_dir / 'app.log',
            level=logging.INFO
        )
        
        # Agent activity logger
        self.agent_logger = self._create_logger(
            'tradingagents.agents',
            self.logs_dir / 'agents.log',
            level=logging.DEBUG,
            json_format=True
        )
        
        # API calls logger
        self.api_logger = self._create_logger(
            'tradingagents.api',
            self.logs_dir / 'api.log',
            level=logging.INFO,
            json_format=True
        )
        
        # Trading activity logger
        self.trading_logger = self._create_logger(
            'tradingagents.trading',
            self.logs_dir / 'trading.log',
            level=logging.INFO,
            json_format=True
        )
        
        # Error logger
        self.error_logger = self._create_logger(
            'tradingagents.errors',
            self.logs_dir / 'errors.log',
            level=logging.ERROR,
            json_format=True
        )
        
        # Security audit logger
        self.audit_logger = self._create_logger(
            'tradingagents.audit',
            self.logs_dir / 'audit.log',
            level=logging.INFO,
            json_format=True
        )
        
        # Performance logger
        self.performance_logger = self._create_logger(
            'tradingagents.performance',
            self.logs_dir / 'performance.log',
            level=logging.INFO,
            json_format=True
        )
    
    def _create_logger(
        self, 
        name: str, 
        log_file: Path, 
        level: int = logging.INFO,
        json_format: bool = False
    ) -> logging.Logger:
        """Create a logger with file and console handlers.
        
        Args:
            name: Logger name
            log_file: Path to log file
            level: Logging level
            json_format: Whether to use JSON formatting
            
        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Set formatters
        if json_format:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add context filter
        file_handler.addFilter(self.context_filter)
        console_handler.addFilter(self.context_filter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        if level <= logging.WARNING:  # Only show warnings and errors on console
            console_handler.setLevel(logging.WARNING)
            logger.addHandler(console_handler)
        
        return logger
    
    @contextmanager
    def context(self, **kwargs):
        """Context manager for setting log context.
        
        Usage:
            with logger.context(agent_id="trader", operation="analyze"):
                logger.info("Performing analysis")
        """
        # Create context from kwargs
        context = LogContext(**kwargs)
        
        # Store old context
        old_context = getattr(self.context_filter.local, 'context', LogContext())
        
        # Set new context
        self.context_filter.set_context(context)
        
        try:
            yield
        finally:
            # Restore old context
            self.context_filter.set_context(old_context)
    
    def log_agent_action(
        self, 
        agent_id: str, 
        action: str, 
        data: Optional[Dict[str, Any]] = None,
        level: int = logging.INFO
    ):
        """Log agent action with context.
        
        Args:
            agent_id: ID of the agent
            action: Action being performed
            data: Additional data to log
            level: Log level
        """
        with self.context(agent_id=agent_id, operation=action):
            self.agent_logger.log(level, f"Agent {agent_id} performing {action}", extra=data or {})
    
    def log_api_call(
        self, 
        endpoint: str, 
        method: str = "GET",
        status_code: Optional[int] = None,
        response_time: Optional[float] = None,
        error: Optional[str] = None
    ):
        """Log API call with details.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            status_code: Response status code
            response_time: Response time in seconds
            error: Error message if any
        """
        extra = {
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'response_time': response_time,
            'error': error
        }
        
        if error:
            self.api_logger.error(f"API call failed: {method} {endpoint}", extra=extra)
        else:
            self.api_logger.info(f"API call: {method} {endpoint}", extra=extra)
    
    def log_trading_action(
        self, 
        action: str, 
        ticker: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        reason: Optional[str] = None
    ):
        """Log trading action.
        
        Args:
            action: Trading action (buy, sell, hold)
            ticker: Stock/asset ticker
            quantity: Trade quantity
            price: Trade price
            reason: Reason for the trade
        """
        extra = {
            'action': action,
            'ticker': ticker,
            'quantity': quantity,
            'price': price,
            'reason': reason
        }
        
        with self.context(ticker=ticker, operation="trade"):
            self.trading_logger.info(f"Trading action: {action} {ticker}", extra=extra)
    
    def log_error(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None
    ):
        """Log error with context.
        
        Args:
            error: Exception that occurred
            context: Additional context information
        """
        extra = context or {}
        extra.update({
            'error_type': type(error).__name__,
            'error_message': str(error)
        })
        
        self.error_logger.error(f"Error occurred: {type(error).__name__}", extra=extra, exc_info=True)
    
    def log_performance(
        self, 
        operation: str, 
        duration: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """Log performance metrics.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            context: Additional context
        """
        extra = context or {}
        extra.update({
            'operation': operation,
            'duration': duration
        })
        
        self.performance_logger.info(f"Performance: {operation} took {duration:.3f}s", extra=extra)
    
    def log_security_event(
        self, 
        event: str, 
        user_id: Optional[str] = None,
        severity: str = "info",
        details: Optional[Dict[str, Any]] = None
    ):
        """Log security audit event.
        
        Args:
            event: Security event description
            user_id: User ID associated with event
            severity: Event severity (info, warning, critical)
            details: Additional event details
        """
        extra = details or {}
        extra.update({
            'event': event,
            'user_id': user_id,
            'severity': severity
        })
        
        level = {
            'info': logging.INFO,
            'warning': logging.WARNING,
            'critical': logging.CRITICAL
        }.get(severity, logging.INFO)
        
        with self.context(user_id=user_id):
            self.audit_logger.log(level, f"Security event: {event}", extra=extra)


# Global logger instance
_logger_instance = None

def get_logger(logs_dir: Optional[Path] = None) -> TradingAgentsLogger:
    """Get or create the global logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = TradingAgentsLogger(logs_dir)
    return _logger_instance

def setup_logging(logs_dir: Optional[Path] = None, debug: bool = False):
    """Set up logging for the application.
    
    Args:
        logs_dir: Directory for log files
        debug: Enable debug logging
    """
    global _logger_instance
    _logger_instance = TradingAgentsLogger(logs_dir)
    
    if debug:
        # Set all loggers to DEBUG level
        for logger_name in ['tradingagents.app', 'tradingagents.agents', 
                           'tradingagents.api', 'tradingagents.trading']:
            logging.getLogger(logger_name).setLevel(logging.DEBUG)