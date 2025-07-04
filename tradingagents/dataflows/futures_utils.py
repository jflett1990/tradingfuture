"""
Futures trading utilities for data fetching and analysis.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

logger = logging.getLogger(__name__)

class FuturesDataError(Exception):
    """Custom exception for futures data errors."""
    pass

def get_futures_symbol(symbol: str) -> str:
    """Convert symbol to Yahoo Finance futures format."""
    if not symbol.endswith("=F"):
        return f"{symbol}=F"
    return symbol

def fetch_futures_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Fetch futures price data from Yahoo Finance.
    
    Args:
        symbol: Futures symbol (e.g., 'CL', 'ES')
        period: Time period for data
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        futures_symbol = get_futures_symbol(symbol)
        ticker = yf.Ticker(futures_symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            raise FuturesDataError(f"No data found for symbol {symbol}")
            
        return data
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        raise FuturesDataError(f"Failed to fetch data for {symbol}")

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) for volatility-based position sizing.
    
    Args:
        data: OHLCV DataFrame
        period: ATR calculation period
        
    Returns:
        ATR values as pandas Series
    """
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr

def analyze_market_structure(data: pd.DataFrame) -> Dict[str, Union[float, str]]:
    """
    Analyze futures market structure (contango/backwardation).
    
    Args:
        data: OHLCV DataFrame
        
    Returns:
        Dictionary with market structure metrics
    """
    if len(data) < 20:
        return {"error": "Insufficient data for analysis"}
    
    current_price = data['Close'].iloc[-1]
    sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
    
    # Fixed: Use ratio for price-agnostic trend analysis
    market_trend = current_price / sma_20 if sma_20 > 0 else 1.0
    
    # Calculate volatility
    returns = data['Close'].pct_change()
    volatility = returns.std() * np.sqrt(252)  # Annualized
    
    # Recent volume trend
    volume_ma = data['Volume'].rolling(window=10).mean()
    current_volume = data['Volume'].iloc[-1]
    volume_ratio = current_volume / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1.0
    
    return {
        "market_trend": market_trend,
        "volatility": volatility,
        "volume_ratio": volume_ratio,
        "current_price": current_price
    }

def calculate_position_size(account_balance: float, risk_per_trade: float, 
                          atr: float, contract_value: float, 
                          atr_multiplier: float = 2.0, 
                          max_position_pct: float = 0.1) -> int:
    """
    Calculate position size based on ATR and risk management.
    
    Args:
        account_balance: Total account balance
        risk_per_trade: Risk percentage per trade (0.01 = 1%)
        atr: Average True Range value
        contract_value: Value per contract
        atr_multiplier: ATR multiplier for stop distance
        max_position_pct: Maximum position as % of account
        
    Returns:
        Number of contracts to trade
    """
    # Input validation
    if atr <= 0 or contract_value <= 0 or account_balance <= 0:
        return 0
    
    # Minimum ATR threshold to prevent division issues
    min_atr = contract_value * 0.001  # 0.1% of contract value
    if atr < min_atr:
        atr = min_atr
    
    # Calculate risk amount and stop distance
    risk_amount = account_balance * risk_per_trade
    stop_distance = atr * atr_multiplier
    
    # Fixed: Correct position sizing using stop distance
    contracts = int(risk_amount / (stop_distance * contract_value))
    
    # Fixed: Apply maximum position size limit
    max_contracts = int((account_balance * max_position_pct) / contract_value)
    contracts = min(contracts, max_contracts)
    
    return max(0, contracts)  # Ensure non-negative

def check_rollover_dates(symbol: str) -> Dict[str, Any]:
    """
    Check if futures contract is approaching expiry.
    
    Args:
        symbol: Futures symbol
        
    Returns:
        Dictionary with rollover information
    """
    # Simplified rollover logic (would need real contract data)
    current_date = datetime.now()
    
    # General rollover patterns (simplified)
    rollover_map = {
        "ES": [3, 6, 9, 12],  # March, June, September, December
        "CL": list(range(1, 13)),  # Monthly
        "GC": [2, 4, 6, 8, 10, 12]  # Bi-monthly
    }
    
    if symbol not in rollover_map:
        return {"days_to_expiry": None, "needs_rollover": False}
    
    months = rollover_map[symbol]
    next_expiry = None
    
    for month in months:
        if month > current_date.month:
            next_expiry = datetime(current_date.year, month, 15)  # Simplified
            break
    
    if next_expiry is None:
        next_expiry = datetime(current_date.year + 1, months[0], 15)
    
    days_to_expiry = (next_expiry - current_date).days
    needs_rollover = days_to_expiry <= 5
    
    return {
        "days_to_expiry": days_to_expiry,
        "needs_rollover": needs_rollover,
        "next_expiry": next_expiry
    }

def validate_trading_hours(symbol: str) -> bool:
    """
    Check if current time is within trading hours for the symbol.
    
    Args:
        symbol: Futures symbol
        
    Returns:
        True if within trading hours
    """
    current_time = datetime.now()
    current_hour = current_time.hour
    
    # Simplified trading hours (EST)
    trading_hours = {
        "ES": (17, 16),  # 5 PM - 4 PM next day (nearly 24/7)
        "CL": (17, 16),  # Similar to ES
        "GC": (17, 16),  # Similar to ES
        "ZC": (19, 13)   # 7 PM - 1:15 PM next day
    }
    
    if symbol not in trading_hours:
        return True  # Default to allowing trading
    
    start_hour, end_hour = trading_hours[symbol]
    
    # Handle overnight sessions
    if start_hour > end_hour:
        return current_hour >= start_hour or current_hour <= end_hour
    else:
        return start_hour <= current_hour <= end_hour