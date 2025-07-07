"""
Monitoring and metrics collection for trading operations.

This module provides comprehensive monitoring capabilities including
performance tracking, system health monitoring, and trading metrics.
"""

import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import threading

logger = logging.getLogger(__name__)


@dataclass
class TradingMetrics:
    """Core trading performance metrics."""
    
    # P&L Metrics
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    
    # Trade Metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    average_win: float = 0.0
    average_loss: float = 0.0
    
    # Performance Ratios
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Risk Metrics
    current_exposure: float = 0.0
    max_position_size: float = 0.0
    var_95: float = 0.0  # Value at Risk
    
    # System Metrics
    uptime: float = 0.0
    error_count: int = 0
    last_update: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize computed fields."""
        if self.last_update is None:
            self.last_update = datetime.now(timezone.utc)
    
    def update_trade_stats(self):
        """Update computed trade statistics."""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
            
        if self.losing_trades > 0 and self.average_loss != 0:
            total_wins = self.winning_trades * self.average_win
            total_losses = abs(self.losing_trades * self.average_loss)
            self.profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        self.last_update = datetime.now(timezone.utc)


@dataclass 
class SystemHealthMetrics:
    """System health and performance metrics."""
    
    # System Performance
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    
    # Network Performance
    latency_ms: float = 0.0
    packet_loss: float = 0.0
    bandwidth_usage: float = 0.0
    
    # Trading System Specific
    market_data_latency: float = 0.0
    order_execution_latency: float = 0.0
    strategy_execution_time: float = 0.0
    
    # Error Rates
    api_error_rate: float = 0.0
    connection_errors: int = 0
    timeout_errors: int = 0
    
    # Timestamps
    last_health_check: Optional[datetime] = None
    
    def __post_init__(self):
        if self.last_health_check is None:
            self.last_health_check = datetime.now(timezone.utc)


class MetricsCollector:
    """Collects and aggregates trading metrics."""
    
    def __init__(self, collection_interval: float = 60.0):
        self.collection_interval = collection_interval
        self.trading_metrics = TradingMetrics()
        self.health_metrics = SystemHealthMetrics()
        
        # Time series data (last 24 hours by default)
        self.pnl_history: deque = deque(maxlen=1440)  # 24 hours of minute data
        self.trade_history: List[Dict[str, Any]] = []
        self.error_history: deque = deque(maxlen=1000)
        
        # Metric collectors
        self._custom_metrics: Dict[str, float] = {}
        self._metric_callbacks: Dict[str, Callable] = {}
        
        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        
    def start_collection(self):
        """Start background metrics collection."""
        if self._running:
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop background metrics collection."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Metrics collection stopped")
    
    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        pnl: float,
        timestamp: Optional[datetime] = None
    ):
        """Record a completed trade."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        with self._lock:
            trade = {
                'timestamp': timestamp,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'pnl': pnl
            }
            
            self.trade_history.append(trade)
            
            # Update trading metrics
            self.trading_metrics.total_trades += 1
            self.trading_metrics.total_pnl += pnl
            self.trading_metrics.realized_pnl += pnl
            self.trading_metrics.daily_pnl += pnl
            
            if pnl > 0:
                self.trading_metrics.winning_trades += 1
                self.trading_metrics.average_win = (
                    (self.trading_metrics.average_win * (self.trading_metrics.winning_trades - 1) + pnl) 
                    / self.trading_metrics.winning_trades
                )
            else:
                self.trading_metrics.losing_trades += 1
                self.trading_metrics.average_loss = (
                    (self.trading_metrics.average_loss * (self.trading_metrics.losing_trades - 1) + pnl)
                    / self.trading_metrics.losing_trades
                )
            
            self.trading_metrics.update_trade_stats()
            
            logger.info(
                f"Trade recorded: {symbol} {side} {quantity} @ {price}, P&L: {pnl:.2f}"
            )
    
    def record_error(
        self,
        error_type: str,
        error_message: str,
        severity: str = "medium",
        context: Optional[Dict[str, Any]] = None
    ):
        """Record a system error."""
        timestamp = datetime.now(timezone.utc)
        
        with self._lock:
            error_record = {
                'timestamp': timestamp,
                'type': error_type,
                'message': error_message,
                'severity': severity,
                'context': context or {}
            }
            
            self.error_history.append(error_record)
            self.trading_metrics.error_count += 1
            
            logger.warning(f"Error recorded: {error_type} - {error_message}")
    
    def update_pnl(self, total_pnl: float, unrealized_pnl: float):
        """Update current P&L values."""
        with self._lock:
            self.trading_metrics.total_pnl = total_pnl
            self.trading_metrics.unrealized_pnl = unrealized_pnl
            self.trading_metrics.realized_pnl = total_pnl - unrealized_pnl
            
            # Add to P&L history
            self.pnl_history.append({
                'timestamp': datetime.now(timezone.utc),
                'total_pnl': total_pnl,
                'unrealized_pnl': unrealized_pnl,
                'realized_pnl': self.trading_metrics.realized_pnl
            })
    
    def update_positions(self, positions: Dict[str, Dict[str, Any]]):
        """Update current position information."""
        with self._lock:
            total_exposure = 0.0
            max_position = 0.0
            
            for symbol, position in positions.items():
                exposure = abs(position.get('market_value', 0.0))
                total_exposure += exposure
                max_position = max(max_position, exposure)
            
            self.trading_metrics.current_exposure = total_exposure
            self.trading_metrics.max_position_size = max_position
    
    def set_custom_metric(self, name: str, value: float):
        """Set a custom metric value."""
        with self._lock:
            self._custom_metrics[name] = value
    
    def register_metric_callback(self, name: str, callback: Callable[[], float]):
        """Register a callback function to compute a metric."""
        self._metric_callbacks[name] = callback
    
    def get_trading_metrics(self) -> TradingMetrics:
        """Get current trading metrics."""
        with self._lock:
            return self.trading_metrics
    
    def get_health_metrics(self) -> SystemHealthMetrics:
        """Get current system health metrics."""
        with self._lock:
            return self.health_metrics
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        with self._lock:
            # Calculate additional metrics
            recent_errors = [
                e for e in self.error_history 
                if (datetime.now(timezone.utc) - e['timestamp']).total_seconds() < 3600
            ]
            
            return {
                'trading_metrics': {
                    'total_pnl': self.trading_metrics.total_pnl,
                    'daily_pnl': self.trading_metrics.daily_pnl,
                    'total_trades': self.trading_metrics.total_trades,
                    'win_rate': self.trading_metrics.win_rate,
                    'profit_factor': self.trading_metrics.profit_factor,
                    'current_exposure': self.trading_metrics.current_exposure,
                },
                'system_health': {
                    'uptime_hours': self.trading_metrics.uptime / 3600,
                    'error_count_total': self.trading_metrics.error_count,
                    'errors_last_hour': len(recent_errors),
                    'market_data_latency': self.health_metrics.market_data_latency,
                    'order_execution_latency': self.health_metrics.order_execution_latency,
                },
                'custom_metrics': self._custom_metrics.copy(),
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
    
    def _collection_loop(self):
        """Background collection loop."""
        start_time = time.time()
        
        while self._running:
            try:
                current_time = time.time()
                self.trading_metrics.uptime = current_time - start_time
                
                # Update callback metrics
                for name, callback in self._metric_callbacks.items():
                    try:
                        value = callback()
                        self._custom_metrics[name] = value
                    except Exception as e:
                        logger.error(f"Error in metric callback {name}: {e}")
                
                # Update health metrics timestamp
                self.health_metrics.last_health_check = datetime.now(timezone.utc)
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(5.0)


class PerformanceTracker:
    """Tracks detailed performance metrics for trading operations."""
    
    def __init__(self):
        self.operation_times: defaultdict = defaultdict(list)
        self.operation_counts: defaultdict = defaultdict(int)
        self._lock = threading.Lock()
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        return _OperationTimer(self, operation_name)
    
    def record_operation_time(self, operation_name: str, duration: float):
        """Record the duration of an operation."""
        with self._lock:
            self.operation_times[operation_name].append(duration)
            self.operation_counts[operation_name] += 1
            
            # Keep only last 1000 measurements per operation
            if len(self.operation_times[operation_name]) > 1000:
                self.operation_times[operation_name] = self.operation_times[operation_name][-1000:]
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, float]:
        """Get statistics for a specific operation."""
        with self._lock:
            times = self.operation_times[operation_name]
            if not times:
                return {}
            
            return {
                'count': len(times),
                'min': min(times),
                'max': max(times),
                'avg': sum(times) / len(times),
                'p50': self._percentile(times, 0.5),
                'p95': self._percentile(times, 0.95),
                'p99': self._percentile(times, 0.99),
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        with self._lock:
            return {
                operation: self.get_operation_stats(operation)
                for operation in self.operation_times.keys()
            }
    
    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(percentile * (len(sorted_data) - 1))
        return sorted_data[index]


class _OperationTimer:
    """Context manager for timing operations."""
    
    def __init__(self, tracker: PerformanceTracker, operation_name: str):
        self.tracker = tracker
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.tracker.record_operation_time(self.operation_name, duration)


class SystemHealthMonitor:
    """Monitors system health and performance."""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self._running = False
        self._thread = None
        self._health_callbacks: List[Callable[[], Dict[str, Any]]] = []
        self._alerts: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def start_monitoring(self):
        """Start health monitoring."""
        if self._running:
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        logger.info("System health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("System health monitoring stopped")
    
    def register_health_check(self, callback: Callable[[], Dict[str, Any]]):
        """Register a health check callback."""
        with self._lock:
            self._health_callbacks.append(callback)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        status = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_status': 'healthy',
            'checks': {},
            'alerts': []
        }
        
        with self._lock:
            # Run all health checks
            for callback in self._health_callbacks:
                try:
                    check_result = callback()
                    status['checks'].update(check_result)
                except Exception as e:
                    logger.error(f"Health check failed: {e}")
                    status['checks']['health_check_error'] = str(e)
                    status['overall_status'] = 'degraded'
            
            # Add recent alerts
            status['alerts'] = list(self._alerts[-10:])  # Last 10 alerts
        
        return status
    
    def add_alert(self, severity: str, message: str, context: Dict[str, Any] = None):
        """Add a system alert."""
        alert = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'severity': severity,
            'message': message,
            'context': context or {}
        }
        
        with self._lock:
            self._alerts.append(alert)
            # Keep only last 100 alerts
            if len(self._alerts) > 100:
                self._alerts = self._alerts[-100:]
        
        logger.warning(f"System alert [{severity}]: {message}")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                status = self.get_health_status()
                
                # Check for critical issues
                if status['overall_status'] != 'healthy':
                    self.add_alert(
                        'warning',
                        'System health degraded',
                        {'status': status['overall_status']}
                    )
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(5.0)