"""
Legacy configuration compatibility layer.

This module provides backward compatibility while encouraging migration
to the new type-safe configuration system.
"""

import os
import warnings
from pathlib import Path

# Import the new configuration system
try:
    from .config import get_legacy_config, get_config
    
    # Issue deprecation warning for direct imports
    warnings.warn(
        "tradingagents.default_config.DEFAULT_CONFIG is deprecated. "
        "Use 'from tradingagents.config import get_config' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Provide the legacy DEFAULT_CONFIG for backward compatibility
    DEFAULT_CONFIG = get_legacy_config()
    
except ImportError:
    # Fallback to original configuration if new system not available
    warnings.warn(
        "New configuration system not available, using legacy configuration. "
        "Install pydantic and pydantic-settings to use the new system.",
        UserWarning,
        stacklevel=2
    )
    
    # Original configuration as fallback
    DEFAULT_CONFIG = {
        "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
        "data_dir": os.getenv("TRADINGAGENTS_DATA_DIR", None),
        "data_cache_dir": os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
            "dataflows/data_cache",
        ),
        # LLM settings
        "llm_provider": "anthropic",
        "deep_think_llm": "claude-3-5-sonnet-20241022",
        "quick_think_llm": "claude-3-5-sonnet-20241022",
        "backend_url": "https://api.anthropic.com/v1",
        # Debate and discussion settings
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "max_recur_limit": 100,
        # Tool settings
        "online_tools": True,
        # Futures trading settings
        "trading_mode": "futures",
        "futures_symbols": {
            "energy": ["CL", "NG", "HO", "RB"],
            "metals": ["GC", "SI", "PL", "PA"],
            "agricultural": ["ZC", "ZS", "ZW", "KC"],
            "financial": ["ES", "NQ", "YM", "RTY"],
            "currencies": ["6E", "6J", "6B", "6A"],
        },
        # Risk management settings for futures
        "max_leverage": 10,
        "margin_buffer": 0.25,
        "position_size_limit": 0.1,
        "rollover_days": 5,
        # Timeframe and trading style settings
        "trading_styles": {
            "swing": {
                "primary_timeframe": "1h",
                "lookback_period": "3mo",
                "target_hold_time": "hours_to_days"
            },
            "scalping": {
                "primary_timeframe": "1m",
                "lookback_period": "1d", 
                "target_hold_time": "seconds_to_minutes",
                "max_hold_minutes": 15,
                "target_ticks": 3.0,
                "stop_ticks": 1.5
            },
            "day_trading": {
                "primary_timeframe": "5m",
                "lookback_period": "5d",
                "target_hold_time": "minutes_to_hours"
            }
        },
        "default_trading_style": "swing"
    }
