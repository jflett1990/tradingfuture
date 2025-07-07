"""
Type-safe configuration management for TradingAgents.

This module provides a Pydantic-based configuration system that ensures
type safety, validation, and proper environment variable handling.
"""

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class TradingStyleConfig(BaseModel):
    """Configuration for specific trading styles."""
    
    primary_timeframe: str = Field(..., description="Primary timeframe for the strategy")
    lookback_period: str = Field(..., description="Lookback period for analysis")
    target_hold_time: str = Field(..., description="Expected holding time")
    max_hold_minutes: Optional[int] = Field(None, description="Maximum hold time in minutes")
    target_ticks: Optional[float] = Field(None, description="Target profit in ticks")
    stop_ticks: Optional[float] = Field(None, description="Stop loss in ticks")


class FuturesSymbolsConfig(BaseModel):
    """Configuration for futures symbols by category."""
    
    energy: List[str] = Field(default=["CL", "NG", "HO", "RB"])
    metals: List[str] = Field(default=["GC", "SI", "PL", "PA"])
    agricultural: List[str] = Field(default=["ZC", "ZS", "ZW", "KC"])
    financial: List[str] = Field(default=["ES", "NQ", "YM", "RTY"])
    currencies: List[str] = Field(default=["6E", "6J", "6B", "6A"])


class RiskManagementConfig(BaseModel):
    """Risk management configuration."""
    
    max_leverage: float = Field(default=10.0, ge=1.0, le=50.0)
    margin_buffer: float = Field(default=0.25, ge=0.0, le=1.0)
    position_size_limit: float = Field(default=0.1, ge=0.01, le=1.0)
    rollover_days: int = Field(default=5, ge=1, le=30)


class TradingAgentsConfig(BaseSettings):
    """Main configuration class for TradingAgents."""
    
    # Directories
    project_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    results_dir: Path = Field(default=Path("./results"))
    data_dir: Optional[Path] = Field(default=None)
    data_cache_dir: Optional[Path] = Field(default=None)
    
    # LLM settings
    llm_provider: Literal["openai", "anthropic", "google"] = Field(default="anthropic")
    deep_think_llm: str = Field(default="claude-3-5-sonnet-20241022")
    quick_think_llm: str = Field(default="claude-3-5-sonnet-20241022")
    backend_url: str = Field(default="https://api.anthropic.com/v1")
    
    # API Keys (from environment)
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    
    # Interactive Brokers
    ib_host: str = Field(default="127.0.0.1", alias="IB_HOST")
    ib_port: int = Field(default=7497, alias="IB_PORT")
    ib_client_id: int = Field(default=1, alias="IB_CLIENT_ID")
    
    # News and data APIs
    newsapi_key: Optional[str] = Field(default=None, alias="NEWSAPI_KEY")
    finnhub_key: Optional[str] = Field(default=None, alias="FINNHUB_KEY")
    
    # Debate and discussion settings
    max_debate_rounds: int = Field(default=1, ge=1, le=10)
    max_risk_discuss_rounds: int = Field(default=1, ge=1, le=10)
    max_recur_limit: int = Field(default=100, ge=10, le=1000)
    
    # Tool settings
    online_tools: bool = Field(default=True)
    
    # Trading settings
    trading_mode: Literal["stocks", "futures"] = Field(default="futures")
    default_trading_style: Literal["swing", "scalping", "day_trading"] = Field(default="swing")
    
    # Futures configuration
    futures_symbols: FuturesSymbolsConfig = Field(default_factory=FuturesSymbolsConfig)
    risk_management: RiskManagementConfig = Field(default_factory=RiskManagementConfig)
    
    # Trading styles
    trading_styles: Dict[str, TradingStyleConfig] = Field(
        default_factory=lambda: {
            "swing": TradingStyleConfig(
                primary_timeframe="1h",
                lookback_period="3mo",
                target_hold_time="hours_to_days"
            ),
            "scalping": TradingStyleConfig(
                primary_timeframe="1m",
                lookback_period="1d",
                target_hold_time="seconds_to_minutes",
                max_hold_minutes=15,
                target_ticks=3.0,
                stop_ticks=1.5
            ),
            "day_trading": TradingStyleConfig(
                primary_timeframe="5m",
                lookback_period="5d",
                target_hold_time="minutes_to_hours"
            )
        }
    )
    
    # Environment
    environment: Literal["development", "testing", "production"] = Field(
        default="development", 
        alias="TRADING_ENV"
    )
    debug: bool = Field(default=False, alias="DEBUG")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "forbid"  # Prevent unknown configuration keys
    
    @validator("data_cache_dir", pre=True, always=True)
    def set_data_cache_dir(cls, v, values):
        """Set default data cache directory if not provided."""
        if v is None:
            project_dir = values.get("project_dir", Path(__file__).parent.parent)
            return project_dir / "tradingagents" / "dataflows" / "data_cache"
        return v
    
    @validator("results_dir", "data_dir", "data_cache_dir", pre=True)
    def resolve_paths(cls, v):
        """Resolve and create directories if they don't exist."""
        if v is not None:
            path = Path(v).resolve()
            path.mkdir(parents=True, exist_ok=True)
            return path
        return v
    
    @validator("backend_url")
    def validate_backend_url(cls, v, values):
        """Validate backend URL matches LLM provider."""
        provider = values.get("llm_provider")
        if provider == "anthropic" and "anthropic.com" not in v:
            raise ValueError("Backend URL must be Anthropic API for anthropic provider")
        elif provider == "openai" and "openai.com" not in v:
            raise ValueError("Backend URL must be OpenAI API for openai provider")
        elif provider == "google" and "googleapis.com" not in v:
            raise ValueError("Backend URL must be Google API for google provider")
        return v
    
    def validate_api_keys(self) -> None:
        """Validate that required API keys are present for the selected provider."""
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when using anthropic provider")
        elif self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when using openai provider")
        elif self.llm_provider == "google" and not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required when using google provider")
    
    def get_current_trading_style(self) -> TradingStyleConfig:
        """Get the configuration for the current trading style."""
        return self.trading_styles[self.default_trading_style]
    
    def to_legacy_dict(self) -> Dict:
        """
        Convert to legacy dictionary format for backward compatibility.
        TODO: Remove this once all code is updated to use the new config system.
        """
        return {
            "project_dir": str(self.project_dir),
            "results_dir": str(self.results_dir),
            "data_dir": str(self.data_dir) if self.data_dir else None,
            "data_cache_dir": str(self.data_cache_dir),
            "llm_provider": self.llm_provider,
            "deep_think_llm": self.deep_think_llm,
            "quick_think_llm": self.quick_think_llm,
            "backend_url": self.backend_url,
            "max_debate_rounds": self.max_debate_rounds,
            "max_risk_discuss_rounds": self.max_risk_discuss_rounds,
            "max_recur_limit": self.max_recur_limit,
            "online_tools": self.online_tools,
            "trading_mode": self.trading_mode,
            "futures_symbols": self.futures_symbols.dict(),
            "max_leverage": self.risk_management.max_leverage,
            "margin_buffer": self.risk_management.margin_buffer,
            "position_size_limit": self.risk_management.position_size_limit,
            "rollover_days": self.risk_management.rollover_days,
            "trading_styles": {
                name: style.dict() for name, style in self.trading_styles.items()
            },
            "default_trading_style": self.default_trading_style,
        }


# Global configuration instance
def get_config() -> TradingAgentsConfig:
    """Get the global configuration instance."""
    return TradingAgentsConfig()


# For backward compatibility, provide the old DEFAULT_CONFIG
def get_legacy_config() -> Dict:
    """Get configuration in legacy dictionary format."""
    return get_config().to_legacy_dict()


# Backward compatibility
DEFAULT_CONFIG = get_legacy_config()