#!/usr/bin/env python3
"""
Autonomous Strategy Validation & Profitability Testing System

This script validates trading strategies across multiple market conditions,
timeframes, and scenarios to prove consistent profitability before live deployment.

Usage:
    python strategy_validator.py --strategy scalping --symbol ES --period 30d
    python strategy_validator.py --comprehensive-test
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
from typing import Dict, List, Any, Optional
import warnings

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import pandas as pd
    import numpy as np
    import yfinance as yf
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.live import Live
    from rich.layout import Layout
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Missing required packages. Install with: pip install pandas numpy yfinance rich matplotlib seaborn")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

console = Console()


class MarketDataProvider:
    """Provides historical market data for backtesting."""
    
    def __init__(self):
        self.cache_dir = Path("data_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_futures_data(self, symbols: List[str], period: str = "6mo", interval: str = "1m") -> Dict[str, pd.DataFrame]:
        """Get historical futures data."""
        console.print(f"[blue]Fetching market data for {symbols}...")
        
        # Futures symbol mapping to ETF proxies for testing
        symbol_mapping = {
            'ES': '^GSPC',  # S&P 500 Index
            'NQ': '^IXIC',  # NASDAQ Index  
            'CL': 'CL=F',   # Crude Oil Futures
            'GC': 'GC=F',   # Gold Futures
            'SI': 'SI=F',   # Silver Futures
            'NG': 'NG=F',   # Natural Gas Futures
        }
        
        data = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for symbol in symbols:
                task = progress.add_task(f"Downloading {symbol}...", total=1)
                
                try:
                    yf_symbol = symbol_mapping.get(symbol, symbol)
                    ticker = yf.Ticker(yf_symbol)
                    
                    # Get historical data
                    hist = ticker.history(period=period, interval=interval)
                    
                    if hist.empty:
                        console.print(f"[red]No data available for {symbol}")
                        continue
                    
                    # Clean and prepare data
                    hist.columns = hist.columns.str.lower()
                    hist = hist.dropna()
                    
                    if len(hist) < 100:
                        console.print(f"[yellow]Insufficient data for {symbol} ({len(hist)} bars)")
                        continue
                    
                    data[symbol] = hist
                    progress.update(task, completed=1)
                    
                except Exception as e:
                    console.print(f"[red]Error fetching data for {symbol}: {e}")
                    continue
        
        console.print(f"[green]Successfully loaded data for {len(data)} symbols")
        return data


class SimpleScalpingStrategy:
    """
    Simple but profitable scalping strategy for validation.
    
    Uses RSI oversold/overbought conditions with volume confirmation
    and tight risk management for quick scalping trades.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_threshold': 1.5,  # Volume must be 1.5x average
            'stop_loss_pct': 0.005,   # 0.5% stop loss
            'take_profit_pct': 0.01,  # 1% take profit
            'position_size': 1.0,     # 1 contract
        }
        self.positions = {}
        self.entry_prices = {}
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        df = data.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config['rsi_period']).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume moving average
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price momentum
        df['price_change'] = df['close'].pct_change()
        
        return df
    
    def generate_signals(self, timestamp: datetime, market_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Generate trading signals based on strategy logic."""
        signals = []
        
        for symbol, data in market_data.items():
            if len(data) < 50:  # Need enough data for indicators
                continue
            
            # Calculate indicators
            df = self.calculate_indicators(data)
            
            if len(df) < 2:
                continue
            
            current = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Check for existing position
            current_position = self.positions.get(symbol, 0)
            
            # Exit signals (risk management)
            if current_position != 0:
                entry_price = self.entry_prices.get(symbol, current['close'])
                
                if current_position > 0:  # Long position
                    # Stop loss or take profit
                    stop_price = entry_price * (1 - self.config['stop_loss_pct'])
                    profit_price = entry_price * (1 + self.config['take_profit_pct'])
                    
                    if current['close'] <= stop_price or current['close'] >= profit_price:
                        signals.append({
                            'symbol': symbol,
                            'action': 'SELL',
                            'quantity': abs(current_position),
                            'price': current['close'],
                            'reason': 'stop_loss' if current['close'] <= stop_price else 'take_profit'
                        })
                        self.positions[symbol] = 0
                        if symbol in self.entry_prices:
                            del self.entry_prices[symbol]
                        continue
                
                elif current_position < 0:  # Short position
                    # Stop loss or take profit
                    stop_price = entry_price * (1 + self.config['stop_loss_pct'])
                    profit_price = entry_price * (1 - self.config['take_profit_pct'])
                    
                    if current['close'] >= stop_price or current['close'] <= profit_price:
                        signals.append({
                            'symbol': symbol,
                            'action': 'BUY',
                            'quantity': abs(current_position),
                            'price': current['close'],
                            'reason': 'stop_loss' if current['close'] >= stop_price else 'take_profit'
                        })
                        self.positions[symbol] = 0
                        if symbol in self.entry_prices:
                            del self.entry_prices[symbol]
                        continue
            
            # Entry signals (only if no position)
            if current_position == 0:
                # Volume confirmation
                volume_confirmed = current['volume_ratio'] >= self.config['volume_threshold']
                
                if not volume_confirmed:
                    continue
                
                # RSI-based signals
                if (current['rsi'] <= self.config['rsi_oversold'] and 
                    previous['rsi'] > self.config['rsi_oversold'] and
                    current['price_change'] > 0):  # Price bouncing up
                    
                    # Long signal
                    signals.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': self.config['position_size'],
                        'price': current['close'],
                        'reason': 'rsi_oversold_bounce'
                    })
                    self.positions[symbol] = self.config['position_size']
                    self.entry_prices[symbol] = current['close']
                
                elif (current['rsi'] >= self.config['rsi_overbought'] and 
                      previous['rsi'] < self.config['rsi_overbought'] and
                      current['price_change'] < 0):  # Price falling
                    
                    # Short signal
                    signals.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': self.config['position_size'],
                        'price': current['close'],
                        'reason': 'rsi_overbought_fall'
                    })
                    self.positions[symbol] = -self.config['position_size']
                    self.entry_prices[symbol] = current['close']
        
        return signals


class StrategyBacktester:
    """Comprehensive strategy backtesting with realistic execution."""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_history = []
        
        # Execution costs
        self.commission_per_contract = 2.50
        self.slippage_bps = 2.0  # 2 basis points
        
        # Contract specifications (simplified)
        self.contract_multipliers = {
            'ES': 50.0,    # $50 per point
            'NQ': 20.0,    # $20 per point
            'CL': 1000.0,  # $1000 per point
            'GC': 100.0,   # $100 per point
            'SI': 5000.0,  # $5000 per point
            'NG': 10000.0, # $10000 per point
        }
    
    def calculate_contract_value(self, symbol: str, price: float, quantity: float) -> float:
        """Calculate contract value including multiplier."""
        multiplier = self.contract_multipliers.get(symbol, 1.0)
        return price * quantity * multiplier
    
    def execute_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade with realistic costs."""
        symbol = signal['symbol']
        action = signal['action']
        quantity = signal['quantity']
        price = signal['price']
        
        # Calculate slippage
        slippage = price * (self.slippage_bps / 10000)
        if action == 'BUY':
            executed_price = price + slippage
        else:
            executed_price = price - slippage
        
        # Calculate commission
        commission = quantity * self.commission_per_contract
        
        # Calculate trade value
        trade_value = self.calculate_contract_value(symbol, executed_price, quantity)
        
        # Update positions
        if symbol not in self.positions:
            self.positions[symbol] = 0
        
        if action == 'BUY':
            self.positions[symbol] += quantity
            self.cash -= commission  # Commission cost
        else:
            self.positions[symbol] -= quantity
            self.cash -= commission  # Commission cost
        
        # Record trade
        trade_record = {
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'intended_price': price,
            'executed_price': executed_price,
            'slippage': abs(executed_price - price),
            'commission': commission,
            'trade_value': trade_value,
            'reason': signal.get('reason', 'unknown')
        }
        
        self.trades.append(trade_record)
        return trade_record
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value."""
        total_value = self.cash
        
        for symbol, quantity in self.positions.items():
            if quantity != 0 and symbol in current_prices:
                position_value = self.calculate_contract_value(symbol, current_prices[symbol], quantity)
                total_value += position_value
        
        return total_value
    
    def run_backtest(self, strategy, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run comprehensive backtest."""
        console.print("[blue]Running strategy backtest...")
        
        # Get all timestamps
        all_timestamps = set()
        for df in market_data.values():
            all_timestamps.update(df.index)
        
        timestamps = sorted(all_timestamps)
        
        # Progress tracking
        total_steps = len(timestamps)
        processed_steps = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Backtesting...", total=total_steps)
            
            for timestamp in timestamps:
                # Prepare market data window
                data_window = {}
                current_prices = {}
                
                for symbol, df in market_data.items():
                    # Get data up to current timestamp
                    mask = df.index <= timestamp
                    window_data = df.loc[mask]
                    
                    if len(window_data) > 0:
                        data_window[symbol] = window_data
                        current_prices[symbol] = window_data['close'].iloc[-1]
                
                # Generate signals
                if len(data_window) > 0:
                    signals = strategy.generate_signals(timestamp, data_window)
                    
                    # Execute signals
                    for signal in signals:
                        self.execute_trade(signal)
                
                # Record portfolio value
                portfolio_value = self.calculate_portfolio_value(current_prices)
                self.equity_history.append({
                    'timestamp': timestamp,
                    'portfolio_value': portfolio_value,
                    'cash': self.cash,
                    'positions': self.positions.copy()
                })
                
                processed_steps += 1
                if processed_steps % 100 == 0:
                    progress.update(task, completed=processed_steps)
            
            progress.update(task, completed=total_steps)
        
        return self.analyze_results()
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze backtest results."""
        if not self.equity_history:
            return {'error': 'No equity history available'}
        
        # Convert to DataFrame for analysis
        equity_df = pd.DataFrame(self.equity_history)
        equity_df['returns'] = equity_df['portfolio_value'].pct_change()
        
        # Calculate metrics
        final_value = equity_df['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Risk metrics
        returns = equity_df['returns'].dropna()
        if len(returns) > 0:
            volatility = returns.std() * np.sqrt(252)  # Annualized
            sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            max_drawdown = self.calculate_max_drawdown(equity_df['portfolio_value'])
        else:
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Trade analysis
        total_trades = len(self.trades)
        if total_trades > 0:
            # Calculate P&L for each trade (simplified)
            profitable_trades = 0
            total_commission = sum(trade['commission'] for trade in self.trades)
            total_slippage = sum(trade['slippage'] for trade in self.trades)
            
            # This is a simplified profit calculation
            for i, trade in enumerate(self.trades):
                if i > 0:  # Compare with previous trade
                    prev_trade = self.trades[i-1]
                    if (trade['action'] == 'SELL' and prev_trade['action'] == 'BUY' and 
                        trade['symbol'] == prev_trade['symbol']):
                        if trade['executed_price'] > prev_trade['executed_price']:
                            profitable_trades += 1
            
            win_rate = profitable_trades / (total_trades / 2) if total_trades > 1 else 0
        else:
            total_commission = 0
            total_slippage = 0
            win_rate = 0
        
        results = {
            'performance': {
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'sharpe_ratio': sharpe_ratio,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100,
            },
            'trading': {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'win_rate_pct': win_rate * 100,
                'total_commission': total_commission,
                'total_slippage': total_slippage,
            },
            'profitability': {
                'is_profitable': total_return > 0,
                'profit_after_costs': final_value - self.initial_capital - total_commission,
                'return_after_costs': (final_value - total_commission - self.initial_capital) / self.initial_capital,
                'meets_criteria': total_return > 0.05 and sharpe_ratio > 1.0 and max_drawdown < 0.15,
            },
            'equity_curve': equity_df,
            'trades': self.trades,
        }
        
        return results
    
    def calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        return abs(drawdown.min())


class StrategyValidator:
    """Main strategy validation orchestrator."""
    
    def __init__(self):
        self.data_provider = MarketDataProvider()
        self.results = {}
    
    def validate_strategy(self, strategy_name: str, symbols: List[str], period: str = "30d") -> Dict[str, Any]:
        """Validate a single strategy."""
        console.print(Panel(f"[bold blue]Validating {strategy_name} Strategy", expand=False))
        
        # Get market data
        market_data = self.data_provider.get_futures_data(symbols, period, interval="5m")
        
        if not market_data:
            console.print("[red]No market data available for validation")
            return {'error': 'No market data'}
        
        # Initialize strategy
        if strategy_name.lower() == 'scalping':
            strategy = SimpleScalpingStrategy()
        else:
            console.print(f"[red]Unknown strategy: {strategy_name}")
            return {'error': f'Unknown strategy: {strategy_name}'}
        
        # Run backtest
        backtester = StrategyBacktester()
        results = backtester.run_backtest(strategy, market_data)
        
        return results
    
    def comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation across multiple strategies and conditions."""
        console.print(Panel("[bold green]Comprehensive Strategy Validation", expand=False))
        
        test_scenarios = [
            {
                'name': 'ES_Scalping_1Week',
                'strategy': 'scalping',
                'symbols': ['ES'],
                'period': '7d',
                'description': 'E-mini S&P 500 scalping over 1 week'
            },
            {
                'name': 'Multi_Symbol_Scalping',
                'strategy': 'scalping', 
                'symbols': ['ES', 'CL', 'GC'],
                'period': '14d',
                'description': 'Multi-symbol scalping over 2 weeks'
            },
            {
                'name': 'Extended_Period_Test',
                'strategy': 'scalping',
                'symbols': ['ES'],
                'period': '30d',
                'description': 'Extended period validation'
            }
        ]
        
        all_results = {}
        
        for scenario in test_scenarios:
            console.print(f"\n[cyan]Running scenario: {scenario['description']}")
            
            try:
                results = self.validate_strategy(
                    scenario['strategy'],
                    scenario['symbols'], 
                    scenario['period']
                )
                
                all_results[scenario['name']] = {
                    'scenario': scenario,
                    'results': results
                }
                
                # Display results
                self.display_results(scenario['name'], results)
                
            except Exception as e:
                console.print(f"[red]Error in scenario {scenario['name']}: {e}")
                all_results[scenario['name']] = {'error': str(e)}
        
        # Summary analysis
        self.display_comprehensive_summary(all_results)
        
        return all_results
    
    def display_results(self, scenario_name: str, results: Dict[str, Any]):
        """Display validation results."""
        if 'error' in results:
            console.print(f"[red]Error: {results['error']}")
            return
        
        perf = results['performance']
        trading = results['trading']
        profit = results['profitability']
        
        # Create results table
        table = Table(title=f"Results: {scenario_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Status", style="green")
        
        # Performance metrics
        table.add_row(
            "Total Return",
            f"{perf['total_return_pct']:.2f}%",
            "✅ PASS" if perf['total_return'] > 0 else "❌ FAIL"
        )
        
        table.add_row(
            "Sharpe Ratio",
            f"{perf['sharpe_ratio']:.2f}",
            "✅ PASS" if perf['sharpe_ratio'] > 1.0 else "❌ FAIL"
        )
        
        table.add_row(
            "Max Drawdown",
            f"{perf['max_drawdown_pct']:.2f}%",
            "✅ PASS" if perf['max_drawdown'] < 0.15 else "❌ FAIL"
        )
        
        table.add_row(
            "Total Trades",
            str(trading['total_trades']),
            "✅ ACTIVE" if trading['total_trades'] > 0 else "⚠️ INACTIVE"
        )
        
        table.add_row(
            "Win Rate",
            f"{trading['win_rate_pct']:.1f}%",
            "✅ GOOD" if trading['win_rate'] > 0.4 else "⚠️ LOW"
        )
        
        table.add_row(
            "Profit After Costs",
            f"${profit['profit_after_costs']:,.2f}",
            "✅ PROFITABLE" if profit['profit_after_costs'] > 0 else "❌ UNPROFITABLE"
        )
        
        table.add_row(
            "Overall Status",
            "MEETS CRITERIA" if profit['meets_criteria'] else "NEEDS IMPROVEMENT",
            "✅ PASS" if profit['meets_criteria'] else "❌ FAIL"
        )
        
        console.print(table)
    
    def display_comprehensive_summary(self, all_results: Dict[str, Any]):
        """Display comprehensive validation summary."""
        console.print("\n" + "="*80)
        console.print(Panel("[bold green]COMPREHENSIVE VALIDATION SUMMARY", expand=False))
        
        passed_scenarios = 0
        total_scenarios = 0
        total_profit = 0
        
        summary_table = Table(title="Scenario Summary")
        summary_table.add_column("Scenario", style="cyan")
        summary_table.add_column("Status", style="bold")
        summary_table.add_column("Return", style="magenta")
        summary_table.add_column("Sharpe", style="blue")
        summary_table.add_column("Drawdown", style="red")
        
        for scenario_name, scenario_data in all_results.items():
            total_scenarios += 1
            
            if 'error' in scenario_data:
                summary_table.add_row(
                    scenario_name,
                    "❌ ERROR",
                    "N/A",
                    "N/A", 
                    "N/A"
                )
                continue
            
            results = scenario_data['results']
            perf = results['performance']
            profit = results['profitability']
            
            status = "✅ PASS" if profit['meets_criteria'] else "❌ FAIL"
            if profit['meets_criteria']:
                passed_scenarios += 1
            
            total_profit += profit['profit_after_costs']
            
            summary_table.add_row(
                scenario_name,
                status,
                f"{perf['total_return_pct']:.2f}%",
                f"{perf['sharpe_ratio']:.2f}",
                f"{perf['max_drawdown_pct']:.2f}%"
            )
        
        console.print(summary_table)
        
        # Overall assessment
        pass_rate = passed_scenarios / total_scenarios if total_scenarios > 0 else 0
        
        console.print(f"\n[bold]OVERALL ASSESSMENT:")
        console.print(f"✅ Passed Scenarios: {passed_scenarios}/{total_scenarios} ({pass_rate:.1%})")
        console.print(f"💰 Total Profit: ${total_profit:,.2f}")
        
        if pass_rate >= 0.8 and total_profit > 0:
            console.print("[bold green]🎉 VALIDATION SUCCESSFUL - STRATEGIES ARE PROFITABLE!")
            console.print("[green]The trading system demonstrates consistent profitability across multiple scenarios.")
            console.print("[green]✅ Ready for paper trading deployment.")
        elif pass_rate >= 0.6:
            console.print("[bold yellow]⚠️ PARTIAL SUCCESS - NEEDS OPTIMIZATION")
            console.print("[yellow]Some strategies show promise but require parameter tuning.")
            console.print("[yellow]🔧 Recommend strategy optimization before deployment.")
        else:
            console.print("[bold red]❌ VALIDATION FAILED - STRATEGIES NEED REWORK")
            console.print("[red]Strategies do not meet profitability criteria.")
            console.print("[red]🚨 Do not deploy with live capital.")


async def main():
    """Main validation orchestrator."""
    parser = argparse.ArgumentParser(description="Autonomous Strategy Validation System")
    parser.add_argument("--strategy", default="scalping", help="Strategy to validate")
    parser.add_argument("--symbol", default="ES", help="Symbol to test")
    parser.add_argument("--period", default="7d", help="Test period")
    parser.add_argument("--comprehensive-test", action="store_true", help="Run comprehensive validation")
    
    args = parser.parse_args()
    
    console.print(Panel("[bold blue]TradingAgents Strategy Validation System", expand=False))
    console.print("[cyan]Autonomous validation of trading strategy profitability")
    
    validator = StrategyValidator()
    
    try:
        if args.comprehensive_test:
            results = validator.comprehensive_validation()
        else:
            symbols = [args.symbol]
            results = validator.validate_strategy(args.strategy, symbols, args.period)
            validator.display_results(f"{args.strategy}_{args.symbol}", results)
        
        # Save results
        results_file = Path("validation_results.json")
        with open(results_file, 'w') as f:
            # Convert DataFrames to dict for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict) and 'results' in value:
                    # Remove DataFrame from results for JSON serialization
                    json_value = value.copy()
                    if 'equity_curve' in json_value['results']:
                        del json_value['results']['equity_curve']
                    json_results[key] = json_value
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, indent=2, default=str)
        
        console.print(f"\n[green]Results saved to {results_file}")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Validation interrupted by user")
    except Exception as e:
        console.print(f"\n[red]Validation failed: {e}")
        logger.error(f"Validation error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())