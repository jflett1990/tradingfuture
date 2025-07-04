"""
Default configuration for the TradingAgents futures trading system.
"""

DEFAULT_CONFIG = {
    "trading_mode": "futures",
    "max_debate_rounds": 2,
    "risk_management": {
        "max_position_pct": 0.1,  # 10% max position size
        "atr_multiplier": 2.0,    # Stop loss multiplier
        "margin_buffer": 0.25,    # 25% margin buffer
        "max_leverage": 10.0,     # Maximum leverage allowed
        "daily_loss_limit": 0.05  # 5% daily loss limit
    },
    "trading_styles": {
        "scalping": {
            "primary_timeframe": "1m",
            "target_ticks": 3.0,
            "stop_ticks": 1.5,
            "max_hold_minutes": 15
        },
        "day_trading": {
            "primary_timeframe": "5m",
            "target_hold_time": "minutes_to_hours"
        },
        "swing": {
            "primary_timeframe": "1h",
            "target_hold_time": "hours_to_days"
        }
    },
    "futures_symbols": {
        "energy": ["CL", "NG", "HO", "RB"],
        "metals": ["GC", "SI", "PL", "PA"],
        "agricultural": ["ZC", "ZS", "ZW", "KC"],
        "financial": ["ES", "NQ", "YM", "RTY"],
        "currency": ["6E", "6J", "6B", "6A"]
    },
    "contract_specs": {
        "ES": {
            "tick_size": 0.25,
            "tick_value": 12.50,
            "contract_size": 50,
            "margin_req": 12000
        },
        "CL": {
            "tick_size": 0.01,
            "tick_value": 10.0,
            "contract_size": 1000,
            "margin_req": 5000
        },
        "GC": {
            "tick_size": 0.10,
            "tick_value": 10.0,
            "contract_size": 100,
            "margin_req": 8000
        }
    },
    "data_sources": {
        "price_data": "yahoo",
        "news_api": "newsapi",
        "fundamental_data": "alpha_vantage"
    },
    "api_keys": {
        "anthropic": None,
        "newsapi": None,
        "alpha_vantage": None
    }
}