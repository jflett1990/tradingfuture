"""
Technical analysis utilities for futures trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI) with proper error handling.
    
    Args:
        data: Price series (typically Close prices)
        period: RSI calculation period
        
    Returns:
        RSI values as pandas Series
    """
    # Fixed: Input validation
    if len(data) < period + 1:
        logger.warning(f"Insufficient data for RSI calculation: {len(data)} < {period + 1}")
        return pd.Series(dtype=float, index=data.index)
    
    if period <= 0:
        raise ValueError("RSI period must be positive")
    
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Fixed: Handle division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        data: Price series
        period: Moving average period
        std_dev: Standard deviation multiplier
        
    Returns:
        Dictionary with upper, middle, and lower bands
    """
    if len(data) < period:
        # Return empty series instead of raising error
        empty_series = pd.Series(dtype=float)
        return {
            "upper": empty_series,
            "middle": empty_series,
            "lower": empty_series
        }
    
    middle_band = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return {
        "upper": upper_band,
        "middle": middle_band,
        "lower": lower_band
    }

def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence) with parameter validation.
    
    Args:
        data: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period
        
    Returns:
        Dictionary with MACD line, signal line, and histogram
    """
    # Fixed: Parameter validation
    if fast >= slow:
        raise ValueError(f"Fast period ({fast}) must be less than slow period ({slow})")
    
    if fast <= 0 or slow <= 0 or signal <= 0:
        raise ValueError("All periods must be positive")
    
    if len(data) < slow + signal:
        logger.warning(f"Insufficient data for MACD: need {slow + signal}, got {len(data)}")
        return {
            "macd": pd.Series(dtype=float, index=data.index), 
            "signal": pd.Series(dtype=float, index=data.index), 
            "histogram": pd.Series(dtype=float, index=data.index)
        }
    
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram
    }

def identify_support_resistance(data: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
    """
    Identify support and resistance levels using local minima/maxima.
    
    Args:
        data: OHLCV DataFrame
        window: Window size for local extrema detection
        
    Returns:
        Dictionary with support and resistance levels
    """
    highs = data['High']
    lows = data['Low']
    
    # Find local maxima (resistance)
    resistance_levels = []
    for i in range(window, len(highs) - window):
        if highs.iloc[i] == highs.iloc[i-window:i+window+1].max():
            resistance_levels.append(highs.iloc[i])
    
    # Find local minima (support)
    support_levels = []
    for i in range(window, len(lows) - window):
        if lows.iloc[i] == lows.iloc[i-window:i+window+1].min():
            support_levels.append(lows.iloc[i])
    
    # Bug #9: Performance issue with large datasets
    # BUG: O(n*w) complexity can be very slow for large datasets
    # BUG: Also doesn't remove duplicate/similar levels
    
    return {
        "resistance": resistance_levels,
        "support": support_levels
    }

def calculate_volatility_bands(data: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
    """
    Calculate volatility-based trading bands.
    
    Args:
        data: Price series
        period: Calculation period
        
    Returns:
        Dictionary with volatility bands
    """
    returns = data.pct_change()
    volatility = returns.rolling(window=period).std()
    
    # Calculate bands based on volatility
    sma = data.rolling(window=period).mean()
    
    # Bug: Incorrect volatility scaling
    # BUG: Not annualizing volatility properly for intraday data
    upper_band = sma + (volatility * data)  # Wrong: should use price scaling
    lower_band = sma - (volatility * data)  # Wrong: should use price scaling
    
    return {
        "upper": upper_band,
        "middle": sma,
        "lower": lower_band,
        "volatility": volatility
    }

def detect_chart_patterns(data: pd.DataFrame, pattern_type: str = "triangle") -> List[Dict]:
    """
    Detect basic chart patterns.
    
    Args:
        data: OHLCV DataFrame
        pattern_type: Type of pattern to detect
        
    Returns:
        List of detected patterns
    """
    patterns = []
    
    if pattern_type == "triangle":
        # Simplified triangle detection
        highs = data['High']
        lows = data['Low']
        
        for i in range(50, len(data) - 10):
            # Look for converging highs and lows
            recent_highs = highs.iloc[i-50:i]
            recent_lows = lows.iloc[i-50:i]
            
            # Check if highs are decreasing and lows are increasing
            high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
            low_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
            
            if high_trend < 0 and low_trend > 0:
                patterns.append({
                    "type": "triangle",
                    "start_index": i-50,
                    "end_index": i,
                    "confidence": 0.6  # Simplified confidence
                })
    
    return patterns

def calculate_momentum_indicators(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate various momentum indicators.
    
    Args:
        data: OHLCV DataFrame
        
    Returns:
        Dictionary of momentum indicators
    """
    close = data['Close']
    high = data['High']
    low = data['Low']
    
    # Rate of Change
    roc_10 = ((close / close.shift(10)) - 1) * 100
    
    # Stochastic Oscillator
    lowest_low = low.rolling(window=14).min()
    highest_high = high.rolling(window=14).max()
    
    # Fixed: Handle division by zero in stochastic calculation
    stoch_range = highest_high - lowest_low
    k_percent = ((close - lowest_low) / stoch_range.replace(0, np.nan)) * 100
    
    d_percent = k_percent.rolling(window=3).mean()
    
    # Williams %R with division by zero protection
    williams_r = ((highest_high - close) / stoch_range.replace(0, np.nan)) * -100
    
    return {
        "roc_10": roc_10,
        "stoch_k": k_percent,
        "stoch_d": d_percent,
        "williams_r": williams_r
    }

def calculate_volume_indicators(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate volume-based indicators.
    
    Args:
        data: OHLCV DataFrame with Volume column
        
    Returns:
        Dictionary of volume indicators
    """
    close = data['Close']
    volume = data['Volume']
    high = data['High']
    low = data['Low']
    
    # On-Balance Volume (OBV)
    obv = pd.Series(index=data.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(data)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    # Volume Price Trend (VPT)
    price_change_pct = close.pct_change()
    vpt = (price_change_pct * volume).cumsum()
    
    # Accumulation/Distribution Line with division by zero protection
    price_range = high - low
    money_flow_multiplier = ((close - low) - (high - close)) / price_range.replace(0, np.nan)
    money_flow_volume = money_flow_multiplier * volume
    ad_line = money_flow_volume.cumsum()
    
    return {
        "obv": obv,
        "vpt": vpt,
        "ad_line": ad_line,
        "money_flow_volume": money_flow_volume
    }

def generate_trading_signals(data: pd.DataFrame, strategy: str = "rsi_bb") -> pd.DataFrame:
    """
    Generate trading signals based on technical indicators.
    
    Args:
        data: OHLCV DataFrame
        strategy: Strategy name
        
    Returns:
        DataFrame with signals
    """
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0  # 0 = hold, 1 = buy, -1 = sell
    
    if strategy == "rsi_bb":
        # RSI + Bollinger Bands strategy
        rsi = calculate_rsi(data['Close'])
        bb = calculate_bollinger_bands(data['Close'])
        
        # Buy signals: RSI oversold and price near lower BB
        buy_condition = (rsi < 30) & (data['Close'] <= bb['lower'] * 1.01)
        
        # Sell signals: RSI overbought and price near upper BB
        sell_condition = (rsi > 70) & (data['Close'] >= bb['upper'] * 0.99)
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        # Add indicator values for reference
        signals['rsi'] = rsi
        signals['bb_upper'] = bb['upper']
        signals['bb_lower'] = bb['lower']
    
    return signals