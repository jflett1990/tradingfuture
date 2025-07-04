import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_dir": "/Users/yluo/Documents/Code/ScAI/FR1-data",
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "claude-3-5-sonnet-20241022",  # Claude 4
    "quick_think_llm": "claude-3-5-sonnet-20241022",  # Claude 4
    "backend_url": "https://api.anthropic.com/v1",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Tool settings
    "online_tools": True,
    # Futures trading settings
    "trading_mode": "futures",  # "stocks" or "futures"
    "futures_symbols": {
        "energy": ["CL", "NG", "HO", "RB"],  # Crude Oil, Natural Gas, Heating Oil, RBOB
        "metals": ["GC", "SI", "PL", "PA"],  # Gold, Silver, Platinum, Palladium
        "agricultural": ["ZC", "ZS", "ZW", "KC"],  # Corn, Soybeans, Wheat, Coffee
        "financial": ["ES", "NQ", "YM", "RTY"],  # E-mini S&P, NASDAQ, Dow, Russell
        "currencies": ["6E", "6J", "6B", "6A"],  # EUR, JPY, GBP, AUD
    },
    # Risk management settings for futures
    "max_leverage": 10,  # Maximum leverage allowed
    "margin_buffer": 0.25,  # 25% buffer above maintenance margin
    "position_size_limit": 0.1,  # Max 10% of portfolio per position
    "rollover_days": 5,  # Days before expiry to consider rollover
    
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
