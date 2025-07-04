# Futures Trading Adaptation Guide

## Overview

This guide describes the adaptation of the TradingAgents framework for futures trading. The original framework was designed for stock trading and has been enhanced to handle the unique characteristics of commodity and financial futures markets.

## Key Adaptations

### 1. Futures-Specific Data Utilities

**New Files:**
- `tradingagents/dataflows/futures_utils.py` - Core futures data fetching and analysis
- `tradingagents/dataflows/futures_technical_utils.py` - Technical analysis for futures

**Key Features:**
- Futures symbol handling (automatic =F suffix)
- Contango/backwardation analysis
- Contract expiry and rollover analysis
- Margin requirement estimates
- Volume and open interest tracking
- Futures curve analysis framework

### 2. Specialized Analyst Agents

**Futures Fundamentals Analyst** (`tradingagents/agents/analysts/futures_fundamentals_analyst.py`)
- Market structure analysis (contango/backwardation)
- Commodity-specific supply/demand factors
- Storage costs and convenience yield
- Seasonal patterns and cyclical factors
- Contract specifications and expiry analysis

**Futures Market Analyst** (`tradingagents/agents/analysts/futures_market_analyst.py`)
- Technical indicators optimized for futures
- Volatility analysis with ATR-based position sizing
- Momentum indicators including CCI (Commodity Channel Index)
- Support/resistance levels for leveraged instruments
- Contract-specific technical considerations

### 3. Futures-Specific Trading Logic

**Futures Trader** (`tradingagents/agents/trader/futures_trader.py`)
- Leverage and margin considerations
- Position sizing for futures contracts
- Contract rollover planning
- Risk management for leveraged positions
- Futures-specific trading strategies

**Futures Risk Manager** (`tradingagents/agents/risk_mgmt/futures_risk_manager.py`)
- Margin requirement analysis
- Leverage risk assessment
- Contract expiry risk evaluation
- Position size validation
- Daily mark-to-market impact
- Risk-adjusted position sizing

### 4. Enhanced Configuration

**Updated Config** (`tradingagents/default_config.py`)
- Trading mode selection (stocks vs futures)
- Futures symbol categorization
- Risk management parameters for futures
- Margin buffer settings
- Rollover timing preferences

## Supported Futures Categories

### Energy Futures
- CL (Crude Oil)
- NG (Natural Gas)
- HO (Heating Oil)
- RB (RBOB Gasoline)

### Metals Futures
- GC (Gold)
- SI (Silver)
- PL (Platinum)
- PA (Palladium)

### Agricultural Futures
- ZC (Corn)
- ZS (Soybeans)
- ZW (Wheat)
- KC (Coffee)

### Financial Futures
- ES (E-mini S&P 500)
- NQ (E-mini NASDAQ)
- YM (E-mini Dow Jones)
- RTY (E-mini Russell 2000)

### Currency Futures
- 6E (Euro)
- 6J (Japanese Yen)
- 6B (British Pound)
- 6A (Australian Dollar)

## Usage

### Command Line Interface

```bash
# Interactive mode
python -m cli.futures_main

# Non-interactive mode
python -m cli.futures_main --symbol CL --date 2024-01-15 --depth standard
```

### Python API

```python
from tradingagents.graph.futures_trading_graph import FuturesTradingGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Create futures trading configuration
config = DEFAULT_CONFIG.copy()
config["trading_mode"] = "futures"
config["max_debate_rounds"] = 2

# Initialize futures trading graph
futures_graph = FuturesTradingGraph(debug=True, config=config)

# Run analysis
state, decision = futures_graph.propagate("CL", "2024-01-15")

print(f"Decision: {decision['risk_decision']}")
print(f"Trader Recommendation: {decision['trader_recommendation']}")
```

## Key Differences from Stock Trading

### 1. Leverage and Margin
- Futures are highly leveraged instruments
- Margin requirements vary by contract and volatility
- Daily mark-to-market settlements
- Maintenance margin requirements

### 2. Contract Specifications
- Standardized contract sizes
- Specific expiry dates
- Physical vs cash settlement
- First notice dates for deliverable contracts

### 3. Market Structure
- Contango vs backwardation
- Basis relationships
- Roll yield considerations
- Seasonal patterns

### 4. Risk Factors
- Higher volatility due to leverage
- Gap risk and limit moves
- Rollover risk near expiry
- Concentration risk in commodity sectors

## Risk Management Framework

### Position Sizing
- Maximum 10% of portfolio per position
- ATR-based position sizing
- Margin buffer requirements (25% above maintenance)
- Leverage limits (max 10x)

### Risk Controls
- Daily mark-to-market monitoring
- Margin requirement tracking
- Expiry date alerts (5 days before)
- Volatility regime assessment

### Stop Loss Strategy
- ATR-based stop losses (typically 2x ATR)
- Futures-specific gap considerations
- Limit move protection
- Rollover timing considerations

## Technical Analysis Adaptations

### Futures-Optimized Indicators
- CCI (Commodity Channel Index) for commodity futures
- ATR for volatility-based position sizing
- Volume analysis for rollover periods
- Momentum indicators for trending markets

### Key Considerations
- Contract month selection
- Volume concentration analysis
- Rollover pattern recognition
- Seasonal adjustment factors

## Fundamental Analysis Framework

### Commodity Futures
- Supply/demand balance
- Inventory levels
- Weather patterns
- Geopolitical factors
- Production/consumption data

### Financial Futures
- Economic indicators
- Interest rate environment
- Currency relationships
- Market sentiment
- Central bank policies

## Implementation Notes

### Data Sources
- Yahoo Finance for price data (yfinance)
- Futures-specific symbol formatting
- Contract specification handling
- Rollover date management

### Memory and Learning
- Futures-specific trading patterns
- Rollover timing optimization
- Risk management lessons
- Market structure insights

## Best Practices

### 1. Pre-Trade Analysis
- Verify contract specifications
- Check margin requirements
- Assess expiry timing
- Evaluate market structure

### 2. Risk Management
- Set position size limits
- Monitor margin requirements
- Plan rollover strategy
- Use appropriate stop losses

### 3. Market Monitoring
- Track volume and open interest
- Monitor basis relationships
- Watch for limit moves
- Assess seasonal patterns

## Limitations and Considerations

### 1. Data Limitations
- Free tier API limitations
- Historical data availability
- Real-time vs delayed data
- Contract specification accuracy

### 2. Risk Disclaimers
- Futures trading involves substantial risk
- Framework is for research/education only
- Not financial advice
- Backtesting required for validation

### 3. Broker Integration
- Margin requirements vary by broker
- Execution capabilities differ
- Commission structures vary
- Technology requirements

## Future Enhancements

### Planned Features
- Real-time margin monitoring
- Advanced rollover strategies
- Multi-contract spread analysis
- Enhanced fundamental data integration

### Integration Opportunities
- Broker API connections
- Real-time data feeds
- Advanced risk metrics
- Portfolio optimization

## Support and Resources

### Documentation
- Original TradingAgents README
- Futures market education resources
- Risk management guides
- Technical analysis resources

### Community
- GitHub issues for bug reports
- Discord for discussions
- Educational webinars
- Trading forums

---

**IMPORTANT DISCLAIMER:** This framework is designed for research and educational purposes only. Futures trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Always consult with qualified financial professionals before making trading decisions.