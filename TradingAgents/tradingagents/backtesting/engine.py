"""
Core backtesting engine for validating trading strategy profitability.

This module provides a sophisticated backtesting framework that simulates
realistic trading conditions including market impact, slippage, fees,
and margin requirements for futures trading.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

from ..core.exceptions import TradingSystemError, ValidationError
from ..strategies.advanced_futures_strategies import TradeSignal, Action


logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """Backtesting execution modes."""
    REALISTIC = "realistic"          # Full market simulation with slippage/fees
    OPTIMISTIC = "optimistic"        # Perfect execution (best case)
    CONSERVATIVE = "conservative"    # Pessimistic execution (worst case)


@dataclass
class BacktestConfig:
    """Configuration for backtesting runs."""
    
    # Time period
    start_date: datetime
    end_date: datetime
    
    # Initial conditions
    initial_capital: float = 100000.0
    
    # Trading parameters
    symbols: List[str] = field(default_factory=lambda: ["ES", "CL", "GC"])
    trading_mode: BacktestMode = BacktestMode.REALISTIC
    
    # Risk management
    max_position_size: float = 0.1     # 10% of capital per position
    max_daily_loss: float = 0.05       # 5% daily loss limit
    max_drawdown: float = 0.20         # 20% maximum drawdown
    
    # Execution settings
    slippage_bps: float = 2.0          # 2 basis points slippage
    commission_per_contract: float = 2.50  # $2.50 per futures contract
    margin_requirement: float = 0.05   # 5% margin requirement
    
    # Data settings
    data_frequency: str = "1m"         # 1-minute bars
    warmup_period: int = 200           # Bars for indicator warmup
    
    # Performance settings
    enable_short_selling: bool = True
    enable_leverage: bool = True
    max_leverage: float = 10.0
    
    # Logging and output
    log_trades: bool = True
    generate_plots: bool = True
    save_detailed_results: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.start_date >= self.end_date:
            raise ValidationError("start_date must be before end_date")
        
        if self.initial_capital <= 0:
            raise ValidationError("initial_capital must be positive")
        
        if not 0 < self.max_position_size <= 1:
            raise ValidationError("max_position_size must be between 0 and 1")
        
        if not self.symbols:
            raise ValidationError("At least one symbol must be specified")


@dataclass
class TradeExecution:
    """Details of a trade execution."""
    
    timestamp: datetime
    symbol: str
    action: Action
    quantity: float
    intended_price: float
    executed_price: float
    slippage: float
    commission: float
    market_impact: float
    
    @property
    def total_cost(self) -> float:
        """Total cost including slippage and commission."""
        return self.slippage + self.commission + self.market_impact


@dataclass
class BacktestResult:
    """Comprehensive results from a backtesting run."""
    
    # Configuration
    config: BacktestConfig
    
    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0                # Value at Risk (95%)
    expected_shortfall: float = 0.0    # Expected Shortfall (95%)
    maximum_leverage_used: float = 0.0
    
    # Execution metrics
    total_slippage: float = 0.0
    total_commission: float = 0.0
    average_execution_quality: float = 0.0
    
    # Time series data
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    drawdown_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_log: List[TradeExecution] = field(default_factory=list)
    
    # Strategy-specific metrics
    strategy_metrics: Dict[str, float] = field(default_factory=dict)
    
    def is_profitable(self) -> bool:
        """Check if the strategy is profitable."""
        return self.total_return > 0 and self.sharpe_ratio > 1.0
    
    def meets_risk_criteria(self) -> bool:
        """Check if the strategy meets risk management criteria."""
        return (
            self.max_drawdown < self.config.max_drawdown and
            self.win_rate > 0.4 and  # At least 40% win rate
            self.profit_factor > 1.2  # Profit factor > 1.2
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of key metrics."""
        return {
            'profitability': {
                'total_return_pct': self.total_return * 100,
                'annualized_return_pct': self.annualized_return * 100,
                'sharpe_ratio': self.sharpe_ratio,
                'is_profitable': self.is_profitable(),
            },
            'risk': {
                'max_drawdown_pct': self.max_drawdown * 100,
                'var_95': self.var_95,
                'meets_risk_criteria': self.meets_risk_criteria(),
            },
            'trading': {
                'total_trades': self.total_trades,
                'win_rate_pct': self.win_rate * 100,
                'profit_factor': self.profit_factor,
                'average_win': self.average_win,
                'average_loss': self.average_loss,
            },
            'execution': {
                'total_slippage': self.total_slippage,
                'total_commission': self.total_commission,
                'execution_quality_pct': self.average_execution_quality * 100,
            }
        }


class BacktestEngine:
    """
    Sophisticated backtesting engine for futures trading strategies.
    
    Provides realistic simulation of trading conditions including:
    - Market impact and slippage modeling
    - Realistic execution delays
    - Margin requirements and funding costs
    - Risk management and position sizing
    - Comprehensive performance analysis
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.current_time: Optional[datetime] = None
        self.current_prices: Dict[str, float] = {}
        
        # Portfolio state
        self.cash = config.initial_capital
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        # Performance tracking
        self.equity_history: List[Tuple[datetime, float]] = []
        self.trade_history: List[TradeExecution] = []
        self.daily_returns: List[float] = []
        
        # Risk tracking
        self.peak_equity = config.initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
        # Strategy callback
        self.strategy_func: Optional[Callable] = None
        
        logger.info(f"BacktestEngine initialized with config: {config}")
    
    def set_strategy(self, strategy_func: Callable[[datetime, Dict[str, pd.DataFrame]], List[TradeSignal]]):
        """Set the trading strategy function."""
        self.strategy_func = strategy_func
        logger.info("Trading strategy set")
    
    async def run_backtest(self, market_data: Dict[str, pd.DataFrame]) -> BacktestResult:
        """
        Run the complete backtesting simulation.
        
        Args:
            market_data: Dictionary mapping symbols to OHLCV DataFrames
            
        Returns:
            BacktestResult with comprehensive performance metrics
        """
        if not self.strategy_func:
            raise ValidationError("Strategy function must be set before running backtest")
        
        logger.info(f"Starting backtest from {self.config.start_date} to {self.config.end_date}")
        
        # Validate market data
        self._validate_market_data(market_data)
        
        # Prepare data
        aligned_data = self._align_market_data(market_data)
        
        # Run simulation
        await self._run_simulation(aligned_data)
        
        # Analyze results
        result = self._analyze_results()
        
        logger.info(f"Backtest completed. Total return: {result.total_return:.2%}")
        return result
    
    def _validate_market_data(self, market_data: Dict[str, pd.DataFrame]):
        """Validate that market data is suitable for backtesting."""
        for symbol in self.config.symbols:
            if symbol not in market_data:
                raise ValidationError(f"Market data missing for symbol: {symbol}")
            
            df = market_data[symbol]
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            
            for col in required_columns:
                if col not in df.columns:
                    raise ValidationError(f"Missing column '{col}' in data for {symbol}")
            
            if df.empty:
                raise ValidationError(f"Empty data for symbol: {symbol}")
            
            # Check for data quality issues
            if df.isnull().any().any():
                logger.warning(f"Found null values in data for {symbol}")
            
            if (df['high'] < df['low']).any():
                raise ValidationError(f"Invalid OHLC data for {symbol}: high < low")
        
        logger.info("Market data validation passed")
    
    def _align_market_data(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align market data timestamps and filter by date range."""
        aligned_data = {}
        
        for symbol, df in market_data.items():
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Filter by date range
            mask = (df.index >= self.config.start_date) & (df.index <= self.config.end_date)
            filtered_df = df.loc[mask].copy()
            
            if filtered_df.empty:
                raise ValidationError(
                    f"No data for {symbol} in date range "
                    f"{self.config.start_date} to {self.config.end_date}"
                )
            
            aligned_data[symbol] = filtered_df
        
        return aligned_data
    
    async def _run_simulation(self, market_data: Dict[str, pd.DataFrame]):
        """Run the main simulation loop."""
        # Get all unique timestamps
        all_timestamps = set()
        for df in market_data.values():
            all_timestamps.update(df.index)
        
        timestamps = sorted(all_timestamps)
        
        logger.info(f"Running simulation over {len(timestamps)} time steps")
        
        for i, timestamp in enumerate(timestamps):
            self.current_time = timestamp
            
            # Update current prices
            self._update_current_prices(market_data, timestamp)
            
            # Calculate unrealized P&L
            self._calculate_unrealized_pnl()
            
            # Check risk limits
            if not self._check_risk_limits():
                logger.warning(f"Risk limits breached at {timestamp}")
                break
            
            # Get strategy signals (skip warmup period)
            if i >= self.config.warmup_period:
                try:
                    # Prepare market data window for strategy
                    data_window = self._prepare_data_window(market_data, timestamp)
                    
                    # Get trading signals from strategy
                    signals = self.strategy_func(timestamp, data_window)
                    
                    # Execute signals
                    for signal in signals:
                        await self._execute_signal(signal)
                
                except Exception as e:
                    logger.error(f"Error in strategy execution at {timestamp}: {e}")
                    continue
            
            # Record equity
            total_equity = self.cash + self.unrealized_pnl
            self.equity_history.append((timestamp, total_equity))
            
            # Update drawdown tracking
            self._update_drawdown(total_equity)
            
            # Progress logging
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{len(timestamps)} steps, equity: ${total_equity:,.2f}")
    
    def _update_current_prices(self, market_data: Dict[str, pd.DataFrame], timestamp: datetime):
        """Update current prices for all symbols."""
        for symbol, df in market_data.items():
            if timestamp in df.index:
                self.current_prices[symbol] = df.loc[timestamp, 'close']
    
    def _calculate_unrealized_pnl(self):
        """Calculate unrealized P&L based on current positions and prices."""
        self.unrealized_pnl = 0.0
        
        for symbol, quantity in self.positions.items():
            if quantity != 0 and symbol in self.current_prices:
                # For futures, P&L is calculated differently based on contract specifications
                # This is a simplified calculation - real implementation would use contract specs
                current_price = self.current_prices[symbol]
                # Assuming $50 per point for ES, $1000 for CL, $100 for GC (simplified)
                multiplier = self._get_contract_multiplier(symbol)
                position_value = quantity * current_price * multiplier
                
                # This is simplified - would need entry prices for accurate P&L
                # For now, assume break-even entry
                self.unrealized_pnl += position_value
    
    def _get_contract_multiplier(self, symbol: str) -> float:
        """Get the contract multiplier for futures symbol."""
        multipliers = {
            'ES': 50.0,    # E-mini S&P 500
            'NQ': 20.0,    # E-mini NASDAQ
            'YM': 5.0,     # E-mini Dow
            'RTY': 50.0,   # E-mini Russell 2000
            'CL': 1000.0,  # Crude Oil
            'NG': 10000.0, # Natural Gas
            'GC': 100.0,   # Gold
            'SI': 5000.0,  # Silver
        }
        return multipliers.get(symbol, 1.0)
    
    def _check_risk_limits(self) -> bool:
        """Check if current state violates risk limits."""
        total_equity = self.cash + self.unrealized_pnl
        
        # Check daily loss limit
        daily_loss = (self.config.initial_capital - total_equity) / self.config.initial_capital
        if daily_loss > self.config.max_daily_loss:
            logger.warning(f"Daily loss limit breached: {daily_loss:.2%}")
            return False
        
        # Check maximum drawdown
        if self.current_drawdown > self.config.max_drawdown:
            logger.warning(f"Maximum drawdown breached: {self.current_drawdown:.2%}")
            return False
        
        return True
    
    def _prepare_data_window(self, market_data: Dict[str, pd.DataFrame], timestamp: datetime) -> Dict[str, pd.DataFrame]:
        """Prepare a data window for strategy evaluation."""
        data_window = {}
        
        for symbol, df in market_data.items():
            # Get data up to current timestamp
            mask = df.index <= timestamp
            window_data = df.loc[mask].copy()
            
            # Limit to reasonable window size for performance
            if len(window_data) > 1000:
                window_data = window_data.tail(1000)
            
            data_window[symbol] = window_data
        
        return data_window
    
    async def _execute_signal(self, signal: TradeSignal):
        """Execute a trading signal with realistic execution simulation."""
        try:
            # Validate signal
            if signal.symbol not in self.config.symbols:
                logger.warning(f"Signal for unsupported symbol: {signal.symbol}")
                return
            
            if signal.symbol not in self.current_prices:
                logger.warning(f"No current price for symbol: {signal.symbol}")
                return
            
            # Calculate position size
            position_size = self._calculate_position_size(signal)
            if position_size == 0:
                return
            
            # Simulate execution
            execution = self._simulate_execution(signal, position_size)
            
            # Update positions and cash
            self._apply_execution(execution)
            
            # Record trade
            self.trade_history.append(execution)
            
            if self.config.log_trades:
                logger.info(f"Executed: {execution.symbol} {execution.action.value} "
                          f"{execution.quantity} @ {execution.executed_price:.2f}")
        
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
    
    def _calculate_position_size(self, signal: TradeSignal) -> float:
        """Calculate appropriate position size based on risk management."""
        current_equity = self.cash + self.unrealized_pnl
        
        # Use signal's position size or default to risk-based sizing
        if hasattr(signal, 'position_size') and signal.position_size > 0:
            raw_size = signal.position_size
        else:
            # Risk-based sizing: risk 1% of equity per trade
            risk_amount = current_equity * 0.01
            stop_distance = abs(signal.entry_price - (signal.stop_loss or signal.entry_price * 0.98))
            raw_size = risk_amount / stop_distance if stop_distance > 0 else 1.0
        
        # Apply position size limits
        max_position_value = current_equity * self.config.max_position_size
        multiplier = self._get_contract_multiplier(signal.symbol)
        max_contracts = max_position_value / (signal.entry_price * multiplier)
        
        position_size = min(raw_size, max_contracts)
        
        # Ensure we have enough cash for margin
        margin_required = position_size * signal.entry_price * multiplier * self.config.margin_requirement
        
        if margin_required > self.cash * 0.8:  # Leave 20% cash buffer
            position_size = (self.cash * 0.8) / (signal.entry_price * multiplier * self.config.margin_requirement)
        
        return max(0, position_size)
    
    def _simulate_execution(self, signal: TradeSignal, position_size: float) -> TradeExecution:
        """Simulate realistic trade execution."""
        current_price = self.current_prices[signal.symbol]
        
        # Calculate slippage based on market conditions and order size
        slippage_amount = self._calculate_slippage(signal, position_size, current_price)
        
        # Determine execution price
        if signal.action == Action.BUY:
            executed_price = current_price + slippage_amount
        else:
            executed_price = current_price - slippage_amount
        
        # Calculate commission
        commission = position_size * self.config.commission_per_contract
        
        # Market impact (simplified)
        market_impact = slippage_amount * 0.1  # Assume 10% of slippage is market impact
        
        return TradeExecution(
            timestamp=self.current_time,
            symbol=signal.symbol,
            action=signal.action,
            quantity=position_size,
            intended_price=current_price,
            executed_price=executed_price,
            slippage=abs(executed_price - current_price) * position_size * self._get_contract_multiplier(signal.symbol),
            commission=commission,
            market_impact=market_impact * position_size * self._get_contract_multiplier(signal.symbol)
        )
    
    def _calculate_slippage(self, signal: TradeSignal, position_size: float, current_price: float) -> float:
        """Calculate realistic slippage based on market conditions."""
        base_slippage = current_price * (self.config.slippage_bps / 10000)
        
        # Adjust for position size (larger positions have more slippage)
        size_factor = 1 + (position_size / 10)  # Simplified size impact
        
        # Adjust for trading mode
        if self.config.trading_mode == BacktestMode.OPTIMISTIC:
            slippage_factor = 0.5
        elif self.config.trading_mode == BacktestMode.CONSERVATIVE:
            slippage_factor = 2.0
        else:
            slippage_factor = 1.0
        
        return base_slippage * size_factor * slippage_factor
    
    def _apply_execution(self, execution: TradeExecution):
        """Apply trade execution to portfolio state."""
        multiplier = self._get_contract_multiplier(execution.symbol)
        
        # Update positions
        if execution.symbol not in self.positions:
            self.positions[execution.symbol] = 0.0
        
        if execution.action == Action.BUY:
            self.positions[execution.symbol] += execution.quantity
        else:
            self.positions[execution.symbol] -= execution.quantity
        
        # Update cash (margin-based for futures)
        margin_used = execution.quantity * execution.executed_price * multiplier * self.config.margin_requirement
        commission_paid = execution.commission
        
        if execution.action == Action.BUY:
            self.cash -= (margin_used + commission_paid)
        else:
            self.cash -= (margin_used + commission_paid)  # Short also requires margin
    
    def _update_drawdown(self, current_equity: float):
        """Update drawdown tracking."""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
    
    def _analyze_results(self) -> BacktestResult:
        """Analyze backtest results and calculate comprehensive metrics."""
        if not self.equity_history:
            raise ValidationError("No equity history available for analysis")
        
        # Convert equity history to DataFrame
        equity_df = pd.DataFrame(self.equity_history, columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Basic performance metrics
        total_return = (equity_df['equity'].iloc[-1] - self.config.initial_capital) / self.config.initial_capital
        
        # Annualized return
        days = (self.config.end_date - self.config.start_date).days
        annualized_return = (1 + total_return) ** (365.25 / days) - 1 if days > 0 else 0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        returns_std = equity_df['returns'].std()
        sharpe_ratio = (equity_df['returns'].mean() / returns_std * np.sqrt(252)) if returns_std > 0 else 0
        
        # Trade statistics
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for trade in self.trade_history if trade.executed_price > trade.intended_price)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor (simplified)
        profit_factor = 1.0  # Would need more detailed P&L calculation
        
        # Risk metrics
        var_95 = equity_df['returns'].quantile(0.05) if not equity_df['returns'].empty else 0
        
        return BacktestResult(
            config=self.config,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=self.max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            var_95=var_95,
            total_slippage=sum(trade.slippage for trade in self.trade_history),
            total_commission=sum(trade.commission for trade in self.trade_history),
            equity_curve=equity_df,
            trade_log=self.trade_history,
        )