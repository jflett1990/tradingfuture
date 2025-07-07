"""
Comprehensive Backtesting Framework for TradingAgents.

This module provides a sophisticated backtesting engine specifically designed
for futures trading strategies with realistic market simulation, slippage
modeling, and comprehensive performance analysis.
"""

from .engine import BacktestEngine, BacktestConfig, BacktestResult
from .portfolio import Portfolio, Position, Trade
from .market_simulator import MarketSimulator, MarketData, OrderBook
from .performance_analyzer import PerformanceAnalyzer, PerformanceMetrics
from .strategy_optimizer import ParameterOptimizer, OptimizationResult
from .risk_analyzer import RiskAnalyzer, RiskMetrics
from .realistic_execution import ExecutionSimulator, SlippageModel, FeeModel

__all__ = [
    # Core backtesting
    "BacktestEngine",
    "BacktestConfig", 
    "BacktestResult",
    
    # Portfolio management
    "Portfolio",
    "Position",
    "Trade",
    
    # Market simulation
    "MarketSimulator",
    "MarketData",
    "OrderBook",
    
    # Performance analysis
    "PerformanceAnalyzer",
    "PerformanceMetrics",
    
    # Strategy optimization
    "ParameterOptimizer",
    "OptimizationResult",
    
    # Risk analysis
    "RiskAnalyzer",
    "RiskMetrics",
    
    # Realistic execution
    "ExecutionSimulator",
    "SlippageModel",
    "FeeModel",
]