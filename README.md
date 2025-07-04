# TradingFuture - AI-Powered Futures Trading System

## 🚀 Overview

An advanced AI-powered futures trading system featuring ultra-fast scalping, multi-timeframe analysis, and real-time news sentiment integration. Built with Claude 4 for enhanced market analysis and Interactive Brokers for live execution.

## ⚡ Key Features

### Multi-Timeframe Trading
- **Scalping Mode**: 1-5 minute charts for range-bound, high-volume periods
- **Day Trading Mode**: 5-15 minute charts for trending intraday moves  
- **Swing Trading Mode**: 1-4 hour charts for longer-term directional moves
- **Adaptive Regime Detection**: Automatically switches between trading styles

### Ultra-Fast Scalping Engine
- Nanosecond-aware execution with prometheus metrics
- Multiple scalping strategies: tick momentum, order flow, news reaction, breakout
- Sub-second signal generation
- Real-time P&L monitoring

### AI-Powered Analysis
- **Claude 4 Integration**: Advanced market analysis and reasoning
- **News Sentiment Analysis**: Lightning-fast news processing with FinBERT
- **Multi-Strategy Consensus**: Combines 4+ strategies with intelligent weighting
- **Risk Management**: Daily P&L caps, position limits, drawdown protection

### Interactive Brokers Integration
- Live market data streaming
- Real-time order execution
- Portfolio and position tracking
- Paper trading and live trading modes

## 🛠 Installation

```bash
git clone https://github.com/jflett1990/tradingfuture.git
cd tradingfuture
pip install -r requirements.txt
```

## 📊 Supported Markets

### Financial Futures
- **E-mini S&P 500** (ES)
- **E-mini NASDAQ** (NQ) 
- **E-mini Dow** (YM)
- **E-mini Russell 2000** (RTY)

### Energy Futures
- **Crude Oil** (CL)
- **Natural Gas** (NG)
- **Heating Oil** (HO)
- **RBOB Gasoline** (RB)

### Metals Futures
- **Gold** (GC)
- **Silver** (SI)
- **Platinum** (PL)
- **Palladium** (PA)

### Agricultural Futures
- **Corn** (ZC)
- **Soybeans** (ZS)
- **Wheat** (ZW)
- **Coffee** (KC)

### Currency Futures
- **Euro** (6E)
- **Japanese Yen** (6J)
- **British Pound** (6B)
- **Australian Dollar** (6A)

## 🎯 Quick Start

### Scalping Mode
```bash
# Start 1-minute scalping on ES
python cli/scalping_cli.py scalp --symbol ES --timeframe 1m --duration 60

# Monitor multiple symbols
python cli/scalping_cli.py monitor --symbols ES,CL,GC
```

### Enhanced Analysis
```bash
# Run full multi-strategy analysis
python -m tradingagents.graph.ib_enhanced_futures_graph ES

# Adaptive timeframe analysis
python -m tradingagents.graph.adaptive_timeframe_graph
```

## 🔧 Configuration

### Environment Variables
```bash
# Interactive Brokers
export IB_HOST="127.0.0.1"
export IB_PORT="7497"  # TWS paper trading

# Anthropic Claude 4
export ANTHROPIC_API_KEY="your_api_key"

# News APIs
export NEWSAPI_KEY="your_newsapi_key"
```

### Trading Configuration
```python
# Located in tradingagents/default_config.py
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
}
```

## 📈 Strategy Components

### Scalping Strategies
1. **Tick Momentum**: Micro momentum detection with volume confirmation
2. **Order Flow**: Level 2 order book imbalance analysis
3. **News Reaction**: Ultra-fast news sentiment scalping
4. **Breakout Scalp**: Micro-breakout detection on small timeframes

### Enhanced Strategies
1. **Lightning News**: Sub-second news reaction trading
2. **Adaptive Momentum**: Multi-timeframe momentum with regime awareness
3. **Volatility Expansion**: Volatility breakout detection
4. **Mean Reversion**: Statistical reversion in ranging markets

## 🛡 Risk Management

- **Daily P&L Caps**: Configurable daily loss limits
- **Position Limits**: Maximum positions per symbol
- **Drawdown Protection**: Circuit breakers for excessive losses
- **Dynamic Sizing**: ATR-based position sizing
- **Margin Management**: Futures margin monitoring

## 📊 Performance Monitoring

### Real-time Metrics
- Live P&L tracking
- Signal generation rates
- Risk utilization percentages
- Market regime detection

### Prometheus Integration
```python
# Metrics exposed on :8500/metrics
- scalp_signals_total
- scalp_ingest_ms
- scalp_latency_ms
```

## 🚨 Important Notes

### Risk Warning
This system trades leveraged futures contracts which carry substantial risk. Past performance does not guarantee future results. Only trade with capital you can afford to lose.

### Paper Trading First
Always test strategies in paper trading mode before deploying live capital.

### IB Requirements
- Interactive Brokers account required for live trading
- TWS or IB Gateway must be running
- Proper permissions for futures trading

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## 🔗 Links

- [Interactive Brokers API](https://www.interactivebrokers.com/en/index.php?f=5041)
- [Claude API Documentation](https://docs.anthropic.com/)
- [Futures Trading Education](https://www.cmegroup.com/education)

---

**⚠️ Disclaimer**: This software is for educational and research purposes. Trading futures involves substantial risk of loss. Users are responsible for their own trading decisions and risk management.