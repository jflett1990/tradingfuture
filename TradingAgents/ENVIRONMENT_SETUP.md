# TradingAgents Environment Setup Guide

This guide explains how to set up and configure TradingAgents with the improved configuration and error handling systems.

## Environment Variables

TradingAgents now supports configuration through environment variables, making it easier to deploy and manage in different environments.

### Required Environment Variables

#### API Keys (at least one required)
```bash
# OpenAI API key
export OPENAI_API_KEY="your_openai_api_key_here"

# Anthropic API key (for Claude models)
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"

# Google API key (for Gemini models)
export GOOGLE_API_KEY="your_google_api_key_here"

# FinnHub API key (for financial data)
export FINNHUB_API_KEY="your_finnhub_api_key_here"
```

### Optional Configuration Variables

#### LLM Configuration
```bash
# LLM Provider (openai, anthropic, google)
export TRADINGAGENTS_LLM_PROVIDER="openai"

# Model selection
export TRADINGAGENTS_DEEP_THINK_MODEL="gpt-4"
export TRADINGAGENTS_QUICK_THINK_MODEL="gpt-3.5-turbo"

# Backend URL (for custom endpoints)
export TRADINGAGENTS_BACKEND_URL="https://api.openai.com/v1"
```

#### Trading Configuration
```bash
# Trading mode
export TRADINGAGENTS_TRADING_MODE="stocks"  # or "futures"

# Risk management
export TRADINGAGENTS_MAX_LEVERAGE="10"

# Online tools usage
export TRADINGAGENTS_ONLINE_TOOLS="true"  # or "false"
```

#### Agent Configuration
```bash
# Debate rounds
export TRADINGAGENTS_MAX_DEBATE_ROUNDS="1"
export TRADINGAGENTS_MAX_RISK_ROUNDS="1"
```

#### Directory Configuration
```bash
# Custom directories
export TRADINGAGENTS_PROJECT_DIR="/path/to/your/project"
export TRADINGAGENTS_RESULTS_DIR="/path/to/results"
export TRADINGAGENTS_DATA_DIR="/path/to/data"
export TRADINGAGENTS_LOGS_DIR="/path/to/logs"
```

#### Security Configuration
```bash
# Audit logging
export TRADINGAGENTS_AUDIT_LOGGING="true"

# API rate limiting
export TRADINGAGENTS_API_RATE_LIMIT="60"
```

## Configuration Files

### Using Configuration Files

You can also use JSON configuration files for more complex setups:

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph

# Load from configuration file
ta = TradingAgentsGraph(debug=True, config_file="config_examples/production.json")
```

### Available Configuration Templates

1. **Development** (`config_examples/development.json`): 
   - Uses cheaper models (gpt-3.5-turbo)
   - Offline tools only
   - Minimal debate rounds
   - Higher rate limits for testing

2. **Production** (`config_examples/production.json`):
   - Uses high-quality models (gpt-4)
   - Online tools enabled
   - Multiple debate rounds for better decisions
   - Conservative rate limits

### Creating Custom Configuration Files

Create a JSON file with your desired settings:

```json
{
  "llm_provider": "openai",
  "deep_think_llm": "gpt-4",
  "quick_think_llm": "gpt-3.5-turbo",
  "backend_url": "https://api.openai.com/v1",
  "trading_mode": "stocks",
  "max_leverage": 5,
  "online_tools": true,
  "max_debate_rounds": 2,
  "max_risk_discuss_rounds": 2,
  "enable_audit_logging": true,
  "max_api_calls_per_minute": 30
}
```

## Setup Examples

### Local Development Setup

1. Create a `.env` file in your project root:
```bash
# .env file
OPENAI_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
TRADINGAGENTS_LLM_PROVIDER=openai
TRADINGAGENTS_DEEP_THINK_MODEL=gpt-3.5-turbo
TRADINGAGENTS_QUICK_THINK_MODEL=gpt-3.5-turbo
TRADINGAGENTS_ONLINE_TOOLS=false
TRADINGAGENTS_MAX_DEBATE_ROUNDS=1
```

2. Load environment variables:
```bash
# Load environment variables from .env file
source .env

# Or use python-dotenv
pip install python-dotenv
```

3. Run TradingAgents:
```python
from tradingagents.graph.trading_graph import TradingAgentsGraph

# Configuration will be loaded automatically from environment
ta = TradingAgentsGraph(debug=True)
```

### Docker Deployment

Create a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  tradingagents:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - FINNHUB_API_KEY=${FINNHUB_API_KEY}
      - TRADINGAGENTS_LLM_PROVIDER=openai
      - TRADINGAGENTS_DEEP_THINK_MODEL=gpt-4
      - TRADINGAGENTS_QUICK_THINK_MODEL=gpt-3.5-turbo
      - TRADINGAGENTS_ONLINE_TOOLS=true
      - TRADINGAGENTS_MAX_DEBATE_ROUNDS=2
      - TRADINGAGENTS_LOGS_DIR=/app/logs
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
```

### Kubernetes Deployment

Create a ConfigMap and Secret:

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: tradingagents-config
data:
  TRADINGAGENTS_LLM_PROVIDER: "openai"
  TRADINGAGENTS_DEEP_THINK_MODEL: "gpt-4"
  TRADINGAGENTS_QUICK_THINK_MODEL: "gpt-3.5-turbo"
  TRADINGAGENTS_ONLINE_TOOLS: "true"
  TRADINGAGENTS_MAX_DEBATE_ROUNDS: "2"

---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: tradingagents-secrets
type: Opaque
data:
  OPENAI_API_KEY: <base64-encoded-key>
  FINNHUB_API_KEY: <base64-encoded-key>
```

## Error Handling and Logging

### Log Files

The improved system creates structured log files:

- `logs/app.log` - Application-level logs
- `logs/agents.log` - Agent activity logs (JSON format)
- `logs/api.log` - API call logs (JSON format)
- `logs/trading.log` - Trading activity logs (JSON format)
- `logs/errors.log` - Error logs (JSON format)
- `logs/audit.log` - Security audit logs (JSON format)
- `logs/performance.log` - Performance metrics (JSON format)

### Log Levels

Control logging verbosity:
```bash
# Enable debug logging
export TRADINGAGENTS_LOG_LEVEL="DEBUG"

# Production logging (default)
export TRADINGAGENTS_LOG_LEVEL="INFO"
```

### Error Handling

The system now provides structured error handling with context:

```python
from tradingagents.utils.exceptions import TradingAgentsError, ConfigurationError

try:
    ta = TradingAgentsGraph(debug=True)
    final_state, decision = ta.propagate("AAPL", "2024-01-01")
except ConfigurationError as e:
    print(f"Configuration issue: {e}")
    print(f"Error code: {e.error_code}")
    print(f"Context: {e.context}")
except TradingAgentsError as e:
    print(f"Trading error: {e}")
    print(f"Context: {e.context}")
```

## Validation and Troubleshooting

### Configuration Validation

The system automatically validates configuration on startup:

```python
from tradingagents.config import get_config_manager

# Validate current configuration
config_manager = get_config_manager()
if config_manager.validate_config():
    print("Configuration is valid")
else:
    print("Configuration validation failed")
```

### Common Issues

1. **Missing API Keys**: Ensure at least one LLM provider API key is set
2. **Invalid Paths**: Check that directory paths exist and are writable
3. **Rate Limits**: Adjust `TRADINGAGENTS_API_RATE_LIMIT` based on your API plan
4. **Model Availability**: Verify that specified models are available for your API key

### Health Checks

You can implement health checks for monitoring:

```python
from tradingagents.config import get_config_manager
from tradingagents.utils.logging_config import get_logger

def health_check():
    try:
        config_manager = get_config_manager()
        config_manager.validate_config()
        
        logger = get_logger()
        logger.log_agent_action("system", "health_check", {"status": "healthy"})
        
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

## Performance Optimization

### Recommended Settings by Use Case

#### High-Frequency Testing
```bash
export TRADINGAGENTS_DEEP_THINK_MODEL="gpt-3.5-turbo"
export TRADINGAGENTS_QUICK_THINK_MODEL="gpt-3.5-turbo"
export TRADINGAGENTS_MAX_DEBATE_ROUNDS="1"
export TRADINGAGENTS_ONLINE_TOOLS="false"
```

#### Production Trading
```bash
export TRADINGAGENTS_DEEP_THINK_MODEL="gpt-4"
export TRADINGAGENTS_QUICK_THINK_MODEL="gpt-3.5-turbo"
export TRADINGAGENTS_MAX_DEBATE_ROUNDS="3"
export TRADINGAGENTS_ONLINE_TOOLS="true"
```

#### Research and Backtesting
```bash
export TRADINGAGENTS_DEEP_THINK_MODEL="gpt-4"
export TRADINGAGENTS_QUICK_THINK_MODEL="gpt-4"
export TRADINGAGENTS_MAX_DEBATE_ROUNDS="5"
export TRADINGAGENTS_ONLINE_TOOLS="true"
```

## Migration from Legacy Configuration

If you're upgrading from the older version:

1. **Environment Variables**: Replace hardcoded config with environment variables
2. **Error Handling**: Wrap your code in try-catch blocks for the new exception types
3. **Logging**: Use the new structured logging system
4. **Configuration Files**: Convert your config dictionaries to JSON files

### Migration Example

**Old way:**
```python
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"
config["deep_think_llm"] = "gpt-4"
ta = TradingAgentsGraph(debug=True, config=config)
```

**New way:**
```bash
export TRADINGAGENTS_LLM_PROVIDER="openai"
export TRADINGAGENTS_DEEP_THINK_MODEL="gpt-4"
```

```python
ta = TradingAgentsGraph(debug=True)  # Configuration loaded automatically
```