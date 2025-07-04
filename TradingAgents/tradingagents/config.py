"""
Configuration management for TradingAgents.

This module provides centralized configuration management with:
- Environment variable support
- Configuration validation
- Type safety
- Default value handling
"""

import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM providers and models."""
    provider: str = "openai"
    deep_think_model: str = "gpt-4"
    quick_think_model: str = "gpt-3.5-turbo"
    backend_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.7


@dataclass
class TradingConfig:
    """Configuration for trading-specific settings."""
    mode: str = "stocks"  # "stocks" or "futures"
    max_leverage: int = 10
    margin_buffer: float = 0.25
    position_size_limit: float = 0.1
    rollover_days: int = 5
    default_style: str = "swing"
    
    futures_symbols: Dict[str, List[str]] = field(default_factory=lambda: {
        "energy": ["CL", "NG", "HO", "RB"],
        "metals": ["GC", "SI", "PL", "PA"],
        "agricultural": ["ZC", "ZS", "ZW", "KC"],
        "financial": ["ES", "NQ", "YM", "RTY"],
        "currencies": ["6E", "6J", "6B", "6A"],
    })
    
    trading_styles: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
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
    })


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""
    project_dir: Path
    results_dir: Path
    data_dir: Path
    data_cache_dir: Path
    logs_dir: Path


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""
    max_debate_rounds: int = 1
    max_risk_discuss_rounds: int = 1
    max_recur_limit: int = 100
    online_tools: bool = True
    enable_memory: bool = True
    memory_retention_days: int = 30


@dataclass
class SecurityConfig:
    """Configuration for security settings."""
    api_key_rotation_days: int = 30
    enable_audit_logging: bool = True
    max_api_calls_per_minute: int = 60
    encrypt_data_at_rest: bool = False


class ConfigManager:
    """Manages application configuration with validation and environment variable support."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_file = config_file
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables and config file."""
        # Start with default project directory
        project_dir = Path(__file__).parent.absolute()
        
        # Load from environment variables
        config = {
            # Path configurations
            "project_dir": self._get_env_path("TRADINGAGENTS_PROJECT_DIR", str(project_dir)),
            "results_dir": self._get_env_path("TRADINGAGENTS_RESULTS_DIR", "./results"),
            "data_dir": self._get_env_path("TRADINGAGENTS_DATA_DIR", "./data"),
            "logs_dir": self._get_env_path("TRADINGAGENTS_LOGS_DIR", "./logs"),
            
            # LLM configurations
            "llm_provider": os.getenv("TRADINGAGENTS_LLM_PROVIDER", "openai"),
            "deep_think_llm": os.getenv("TRADINGAGENTS_DEEP_THINK_MODEL", "gpt-4"),
            "quick_think_llm": os.getenv("TRADINGAGENTS_QUICK_THINK_MODEL", "gpt-3.5-turbo"),
            "backend_url": os.getenv("TRADINGAGENTS_BACKEND_URL", "https://api.openai.com/v1"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
            "google_api_key": os.getenv("GOOGLE_API_KEY"),
            
            # Trading configurations
            "trading_mode": os.getenv("TRADINGAGENTS_TRADING_MODE", "stocks"),
            "max_leverage": int(os.getenv("TRADINGAGENTS_MAX_LEVERAGE", "10")),
            "online_tools": os.getenv("TRADINGAGENTS_ONLINE_TOOLS", "true").lower() == "true",
            
            # Agent configurations
            "max_debate_rounds": int(os.getenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "1")),
            "max_risk_discuss_rounds": int(os.getenv("TRADINGAGENTS_MAX_RISK_ROUNDS", "1")),
            
            # Security configurations
            "enable_audit_logging": os.getenv("TRADINGAGENTS_AUDIT_LOGGING", "true").lower() == "true",
            "max_api_calls_per_minute": int(os.getenv("TRADINGAGENTS_API_RATE_LIMIT", "60")),
        }
        
        # Calculate derived paths
        config["data_cache_dir"] = config["project_dir"] / "dataflows" / "data_cache"
        
        # Load from config file if provided
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config)
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {self.config_file}: {e}")
        
        return config
    
    def _get_env_path(self, env_var: str, default: str) -> Path:
        """Get path from environment variable with default."""
        value = os.getenv(env_var, str(default))
        return Path(value).expanduser().absolute()
    
    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration."""
        return LLMConfig(
            provider=self._config["llm_provider"],
            deep_think_model=self._config["deep_think_llm"],
            quick_think_model=self._config["quick_think_llm"],
            backend_url=self._config["backend_url"],
            api_key=self._get_api_key(),
        )
    
    def get_trading_config(self) -> TradingConfig:
        """Get trading configuration."""
        return TradingConfig(
            mode=self._config["trading_mode"],
            max_leverage=self._config["max_leverage"],
        )
    
    def get_path_config(self) -> PathConfig:
        """Get path configuration."""
        return PathConfig(
            project_dir=self._config["project_dir"],
            results_dir=self._config["results_dir"],
            data_dir=self._config["data_dir"],
            data_cache_dir=self._config["data_cache_dir"],
            logs_dir=self._config["logs_dir"],
        )
    
    def get_agent_config(self) -> AgentConfig:
        """Get agent configuration."""
        return AgentConfig(
            max_debate_rounds=self._config["max_debate_rounds"],
            max_risk_discuss_rounds=self._config["max_risk_discuss_rounds"],
            online_tools=self._config["online_tools"],
        )
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        return SecurityConfig(
            enable_audit_logging=self._config["enable_audit_logging"],
            max_api_calls_per_minute=self._config["max_api_calls_per_minute"],
        )
    
    def _get_api_key(self) -> Optional[str]:
        """Get appropriate API key based on provider."""
        provider = self._config["llm_provider"].lower()
        if provider == "openai":
            return self._config.get("openai_api_key")
        elif provider == "anthropic":
            return self._config.get("anthropic_api_key")
        elif provider == "google":
            return self._config.get("google_api_key")
        return None
    
    def validate_config(self) -> bool:
        """Validate configuration values."""
        errors = []
        
        # Validate API key
        api_key = self._get_api_key()
        if not api_key:
            errors.append(f"Missing API key for provider: {self._config['llm_provider']}")
        
        # Validate paths
        try:
            for path_key in ["project_dir", "results_dir", "data_dir", "data_cache_dir", "logs_dir"]:
                path = self._config[path_key]
                if not isinstance(path, Path):
                    errors.append(f"Invalid path for {path_key}: {path}")
        except Exception as e:
            errors.append(f"Path validation error: {e}")
        
        # Validate numeric values
        numeric_configs = {
            "max_leverage": (1, 100),
            "max_debate_rounds": (1, 10),
            "max_risk_discuss_rounds": (1, 10),
            "max_api_calls_per_minute": (1, 1000),
        }
        
        for key, (min_val, max_val) in numeric_configs.items():
            value = self._config.get(key, 0)
            if not isinstance(value, int) or value < min_val or value > max_val:
                errors.append(f"Invalid {key}: {value} (should be between {min_val} and {max_val})")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def create_directories(self) -> None:
        """Create necessary directories."""
        path_config = self.get_path_config()
        directories = [
            path_config.results_dir,
            path_config.data_dir,
            path_config.data_cache_dir,
            path_config.logs_dir,
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                raise
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format for backward compatibility."""
        return self._config.copy()
    
    def save_config(self, file_path: str) -> None:
        """Save current configuration to file."""
        config_copy = self._config.copy()
        
        # Convert Path objects to strings for JSON serialization
        for key, value in config_copy.items():
            if isinstance(value, Path):
                config_copy[key] = str(value)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(config_copy, f, indent=2)
            logger.info(f"Configuration saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {e}")
            raise


# Global configuration manager instance
_config_manager = None

def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """Get or create the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager

def get_default_config() -> Dict[str, Any]:
    """Get default configuration as dictionary for backward compatibility."""
    manager = get_config_manager()
    return manager.to_dict()