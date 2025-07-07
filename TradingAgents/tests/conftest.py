"""
Pytest configuration and fixtures for TradingAgents testing.

This module provides shared fixtures, test configuration, and utilities
for testing all components of the trading system.
"""

import asyncio
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import Mock, AsyncMock, patch

import pytest
import pandas as pd
import numpy as np
from ib_insync import Contract, Stock, Future

from tradingagents.config import TradingAgentsConfig, get_config
from tradingagents.strategies.advanced_futures_strategies import TradeSignal, Action


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration with temporary directories."""
    # Set environment variables for testing
    test_env = {
        "TRADING_ENV": "testing",
        "DEBUG": "true",
        "ANTHROPIC_API_KEY": "test_key_123",
        "IB_HOST": "127.0.0.1",
        "IB_PORT": "7497",
        "MAX_DAILY_LOSS": "1000.0",
        "MAX_POSITIONS_PER_SYMBOL": "2",
        "PAPER_TRADING_MODE": "true",
    }
    
    with patch.dict(os.environ, test_env):
        config = TradingAgentsConfig(
            project_dir=temp_dir,
            results_dir=temp_dir / "results",
            data_cache_dir=temp_dir / "data_cache",
        )
        yield config


@pytest.fixture
def mock_ib_client():
    """Create a mock Interactive Brokers client."""
    mock_client = Mock()
    mock_client.isConnected.return_value = True
    mock_client.reqHistoricalData = Mock()
    mock_client.reqMktData = Mock()
    mock_client.placeOrder = Mock()
    mock_client.cancelOrder = Mock()
    mock_client.portfolio = []
    mock_client.positions = []
    mock_client.orders = []
    return mock_client


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing."""
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = Mock(content="Mock LLM response for testing")
    mock_llm.astream.return_value = AsyncMock()
    return mock_llm


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=30),
        end=datetime.now(),
        freq='H'
    )
    
    # Generate realistic OHLCV data
    np.random.seed(42)  # For reproducible tests
    base_price = 4000.0
    returns = np.random.normal(0.0001, 0.02, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.normal(0, 1, len(prices)),
        'high': prices + np.abs(np.random.normal(5, 2, len(prices))),
        'low': prices - np.abs(np.random.normal(5, 2, len(prices))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(prices)),
    })
    
    return data


@pytest.fixture
def sample_futures_contracts():
    """Generate sample futures contracts for testing."""
    contracts = {
        'ES': Future(symbol='ES', lastTradeDateOrContractMonth='20241220', exchange='CME'),
        'CL': Future(symbol='CL', lastTradeDateOrContractMonth='20241220', exchange='NYMEX'),
        'GC': Future(symbol='GC', lastTradeDateOrContractMonth='20241227', exchange='COMEX'),
        'NQ': Future(symbol='NQ', lastTradeDateOrContractMonth='20241220', exchange='CME'),
    }
    return contracts


@pytest.fixture
def sample_trade_signals():
    """Generate sample trade signals for testing."""
    signals = [
        TradeSignal(
            symbol="ES",
            action=Action.BUY,
            confidence=0.85,
            entry_price=4000.0,
            stop_loss=3995.0,
            take_profit=4010.0,
            position_size=1.0,
            strategy="test_strategy",
            reasoning="Test reasoning for buy signal",
            risk_reward_ratio=2.0,
            urgency="medium",
            news_driven=False,
            timestamp=datetime.now(timezone.utc)
        ),
        TradeSignal(
            symbol="CL",
            action=Action.SELL,
            confidence=0.72,
            entry_price=70.50,
            stop_loss=71.00,
            take_profit=69.50,
            position_size=2.0,
            strategy="scalping",
            reasoning="Short-term bearish momentum",
            risk_reward_ratio=2.0,
            urgency="high",
            news_driven=True,
            timestamp=datetime.now(timezone.utc)
        )
    ]
    return signals


@pytest.fixture
def mock_news_data():
    """Generate mock news data for testing."""
    news_items = [
        {
            'title': 'Federal Reserve announces interest rate decision',
            'content': 'The Federal Reserve has decided to maintain current interest rates...',
            'timestamp': datetime.now(timezone.utc) - timedelta(hours=1),
            'source': 'Reuters',
            'sentiment': 0.1,  # Slightly positive
            'urgency': 0.8,
            'symbols': ['ES', 'NQ']
        },
        {
            'title': 'Oil prices surge on supply concerns',
            'content': 'Crude oil prices jumped 3% in early trading...',
            'timestamp': datetime.now(timezone.utc) - timedelta(minutes=30),
            'source': 'Bloomberg',
            'sentiment': 0.6,  # Positive for oil
            'urgency': 0.9,
            'symbols': ['CL']
        },
        {
            'title': 'Gold reaches new highs amid uncertainty',
            'content': 'Gold futures continue their upward trajectory...',
            'timestamp': datetime.now(timezone.utc) - timedelta(minutes=15),
            'source': 'CNBC',
            'sentiment': 0.4,
            'urgency': 0.7,
            'symbols': ['GC']
        }
    ]
    return news_items


@pytest.fixture
def mock_portfolio_state():
    """Generate mock portfolio state for testing."""
    return {
        'cash': 100000.0,
        'positions': {
            'ES': {'quantity': 2, 'avg_cost': 3995.0, 'market_value': 8000.0},
            'CL': {'quantity': -1, 'avg_cost': 70.75, 'market_value': -7050.0}
        },
        'total_value': 100950.0,
        'unrealized_pnl': 950.0,
        'realized_pnl': 0.0,
        'daily_pnl': 950.0
    }


@pytest.fixture
def mock_risk_metrics():
    """Generate mock risk metrics for testing."""
    return {
        'var_95': 2500.0,  # 95% Value at Risk
        'max_drawdown': 0.05,  # 5% maximum drawdown
        'sharpe_ratio': 1.2,
        'volatility': 0.15,  # 15% annualized volatility
        'beta': 1.1,
        'correlation_matrix': {
            'ES': {'ES': 1.0, 'NQ': 0.85, 'CL': 0.2},
            'NQ': {'ES': 0.85, 'NQ': 1.0, 'CL': 0.15},
            'CL': {'ES': 0.2, 'NQ': 0.15, 'CL': 1.0}
        }
    }


@pytest.fixture
def mock_order_book():
    """Generate mock Level 2 order book data."""
    return {
        'ES': {
            'timestamp': datetime.now(timezone.utc),
            'bids': [
                {'price': 3999.75, 'size': 50},
                {'price': 3999.50, 'size': 75},
                {'price': 3999.25, 'size': 100},
                {'price': 3999.00, 'size': 125},
                {'price': 3998.75, 'size': 150}
            ],
            'asks': [
                {'price': 4000.00, 'size': 45},
                {'price': 4000.25, 'size': 80},
                {'price': 4000.50, 'size': 95},
                {'price': 4000.75, 'size': 110},
                {'price': 4001.00, 'size': 140}
            ]
        }
    }


# Test markers for different test categories
pytest_plugins = []

# Custom test markers
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]


class TestBase:
    """Base class for all test classes providing common utilities."""
    
    @staticmethod
    def assert_valid_trade_signal(signal: TradeSignal):
        """Assert that a trade signal is valid."""
        assert signal.symbol is not None
        assert signal.action in [Action.BUY, Action.SELL]
        assert 0.0 <= signal.confidence <= 1.0
        assert signal.entry_price > 0
        assert signal.position_size > 0
        assert signal.strategy is not None
        assert signal.timestamp is not None
        
        # Risk/reward validation
        if signal.take_profit and signal.stop_loss:
            if signal.action == Action.BUY:
                assert signal.take_profit > signal.entry_price
                assert signal.stop_loss < signal.entry_price
            else:
                assert signal.take_profit < signal.entry_price
                assert signal.stop_loss > signal.entry_price
    
    @staticmethod
    def assert_portfolio_consistency(portfolio_state: Dict):
        """Assert that portfolio state is consistent."""
        assert 'cash' in portfolio_state
        assert 'positions' in portfolio_state
        assert 'total_value' in portfolio_state
        assert portfolio_state['cash'] >= 0  # No negative cash in testing
        
        # Validate positions
        for symbol, position in portfolio_state['positions'].items():
            assert 'quantity' in position
            assert 'avg_cost' in position
            assert 'market_value' in position
            assert position['avg_cost'] > 0


# Async test utilities
class AsyncTestBase:
    """Base class for async tests."""
    
    @pytest.fixture(autouse=True)
    def setup_async_test(self, event_loop):
        """Setup for async tests."""
        self.loop = event_loop
    
    async def wait_for_condition(self, condition_func, timeout=5.0, interval=0.1):
        """Wait for a condition to become true with timeout."""
        start_time = datetime.now().timestamp()
        while datetime.now().timestamp() - start_time < timeout:
            if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
                return True
            await asyncio.sleep(interval)
        return False