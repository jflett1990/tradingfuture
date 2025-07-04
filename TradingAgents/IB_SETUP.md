# Interactive Brokers Integration Setup Guide

This guide will help you set up the Interactive Brokers (IB) integration for live futures trading.

## Prerequisites

1. **Interactive Brokers Account**
   - Live trading account or paper trading account
   - Account approved for futures trading
   - TWS (Trader Workstation) or IB Gateway installed

2. **Anthropic API Access**
   - Anthropic API key for Claude 4 (claude-3-5-sonnet-20241022)
   - Sufficient API credits for analysis requests

3. **Python Environment**
   - Python 3.8+ (recommended 3.10+)
   - Virtual environment (recommended)

## Installation Steps

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv_ib
source venv_ib/bin/activate  # On Windows: venv_ib\Scripts\activate

# Install IB-specific requirements
pip install -r requirements_ib.txt

# Install TA-Lib (may require additional setup)
# On macOS: brew install ta-lib
# On Ubuntu: sudo apt-get install libta-lib-dev
# On Windows: Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
```

### 2. Configure Interactive Brokers

#### TWS Configuration
1. **Start TWS or IB Gateway**
   - TWS: Full trading platform
   - IB Gateway: Lightweight API-only interface (recommended for automated trading)

2. **Enable API Access**
   - In TWS: File → Global Configuration → API → Settings
   - Check "Enable ActiveX and Socket Clients"
   - Check "Allow connections from localhost only" (for security)
   - Socket port: 7497 (paper trading) or 7496 (live trading)
   - Trusted IP addresses: 127.0.0.1

3. **API Settings**
   - Master API client ID: Leave blank or set to your preference
   - Read-Only API: Uncheck (to allow trading)
   - Download open orders on connection: Check (recommended)

#### Paper Trading Setup (Recommended for Testing)
1. **Paper Trading Account**
   - Create paper trading account in Account Management
   - Login to TWS with paper trading credentials
   - Use port 7497 for paper trading

2. **Test Connection**
   ```bash
   # Test basic IB connection
   python -c "
   import asyncio
   from tradingagents.brokers.ib_broker import create_ib_broker
   
   async def test():
       broker = await create_ib_broker(paper_trading=True)
       print('✅ IB Connection successful!')
       await broker.close()
   
   asyncio.run(test())
   "
   ```

### 3. Claude 4 Configuration

The system is configured to use Claude 4 (claude-3-5-sonnet-20241022) for all AI analysis:

1. **Get Anthropic API Key**
   - Sign up at https://console.anthropic.com/
   - Create an API key with sufficient credits
   - Store securely (never commit to version control)

2. **Benefits of Claude 4**
   - Superior reasoning for complex market analysis
   - Better understanding of futures trading nuances
   - More accurate risk assessment
   - Enhanced strategy reasoning

### 4. Environment Configuration

Create a `.env` file in your project root:

```bash
# Interactive Brokers Settings
IB_HOST=127.0.0.1
IB_PORT=7497              # 7497 for paper, 7496 for live
IB_CLIENT_ID=1            # Unique client ID

# Risk Management
IB_MAX_POSITION_VALUE=100000   # Maximum position value in USD
IB_MAX_DAILY_LOSS=5000         # Maximum daily loss in USD
IB_MAX_OPEN_ORDERS=20          # Maximum concurrent orders

# Trading Settings
IB_PAPER_TRADING=true          # Set to false for live trading
IB_ENABLE_STREAMING=true       # Enable real-time data

# Anthropic API (for Claude 4 analysis)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 5. Verify Installation

Run the verification script:

```bash
# Test paper trading setup
python -m tradingagents.brokers.ib_broker

# Test CLI interface
python cli/ib_futures_cli.py wizard

# Test full integration
python -m tradingagents.graph.ib_enhanced_futures_graph --symbol CL
```

## Usage Examples

### 1. Interactive CLI (Recommended)

```bash
# Paper trading wizard
python cli/ib_futures_cli.py wizard

# Live trading wizard (requires confirmation)
python cli/ib_futures_cli.py wizard --live

# Quick analysis
python cli/ib_futures_cli.py trade --symbol CL --action analyze

# Portfolio view
python cli/ib_futures_cli.py portfolio --live
```

### 2. Programmatic Usage

```python
import asyncio
from tradingagents.graph.ib_enhanced_futures_graph import create_ib_enhanced_graph

async def analyze_crude_oil():
    # Create IB-enhanced graph
    graph = await create_ib_enhanced_graph(paper_trading=True)
    
    # Run analysis
    decision = await graph.propagate("CL", "2025-01-15")
    
    print(f"Decision: {decision['action']}")
    print(f"Confidence: {decision['confidence']:.1%}")
    print(f"Live Price: ${decision['live_price']}")
    print(f"Position Size: {decision['position_size']}")

# Run analysis
asyncio.run(analyze_crude_oil())
```

### 3. Live Trading Example

```python
import asyncio
from tradingagents.brokers.ib_broker import create_ib_broker
from tradingagents.strategies.advanced_strategies import TradeSignal, Action

async def execute_live_trade():
    # ⚠️ LIVE TRADING - REAL MONEY AT RISK ⚠️
    broker = await create_ib_broker(paper_trading=False)
    
    # Create trade signal
    signal = TradeSignal(
        symbol="CL",
        action=Action.BUY,
        confidence=0.85,
        entry_price=75.50,
        stop_loss=74.00,
        take_profit=78.00,
        position_size=1.0,
        strategy="manual",
        reasoning="Manual test trade",
        r_r_ratio=1.67,
        urgency="medium",
        news_catalyst=False
    )
    
    # Execute trade
    trade = await broker.execute_signal(signal)
    if trade:
        print(f"✅ Trade executed: Order ID {trade.order.orderId}")
    else:
        print("❌ Trade rejected by risk management")
    
    await broker.close()

# ⚠️ Uncomment only for live trading
# asyncio.run(execute_live_trade())
```

## Supported Futures Contracts

The system supports the following futures contracts:

### Energy
- **CL**: Crude Oil (NYMEX)
- **NG**: Natural Gas (NYMEX)
- **HO**: Heating Oil (NYMEX)
- **RB**: RBOB Gasoline (NYMEX)

### Metals
- **GC**: Gold (COMEX)
- **SI**: Silver (COMEX)
- **HG**: Copper (COMEX)
- **PA**: Palladium (NYMEX)

### Agriculture
- **ZC**: Corn (CBOT)
- **ZS**: Soybeans (CBOT)
- **ZW**: Wheat (CBOT)
- **KC**: Coffee (ICE)

### Financial
- **ES**: E-mini S&P 500 (CME)
- **NQ**: E-mini NASDAQ-100 (CME)
- **YM**: E-mini Dow Jones (CBOT)
- **RTY**: E-mini Russell 2000 (CME)

### Currencies
- **6E**: Euro (CME)
- **6B**: British Pound (CME)
- **6J**: Japanese Yen (CME)
- **6A**: Australian Dollar (CME)

## Risk Management Features

The system includes comprehensive risk management:

1. **Position Limits**
   - Maximum position value per trade
   - Maximum percentage of account at risk
   - Position size scaling based on volatility

2. **Daily Limits**
   - Maximum daily loss protection
   - Maximum number of open orders
   - Circuit breakers for unusual market conditions

3. **Signal Validation**
   - Confidence thresholds
   - Risk/reward ratio requirements
   - News catalyst consideration

4. **Real-time Monitoring**
   - Live P&L tracking
   - Position monitoring
   - Connection status alerts

## Monitoring and Metrics

The system provides Prometheus metrics on port 9000:

- `ib_orders_total`: Total orders placed
- `ib_fills_total`: Total order fills
- `ib_unrealized_pnl`: Current unrealized P&L

Access metrics at: http://localhost:9000

## Troubleshooting

### Common Issues

1. **Connection Failed**
   ```
   Error: Failed to connect to IB
   ```
   - Ensure TWS/Gateway is running
   - Check API is enabled in TWS settings
   - Verify port number (7497 for paper, 7496 for live)
   - Check firewall settings

2. **Contract Not Found**
   ```
   Error: Unsupported futures symbol
   ```
   - Verify symbol is in supported list
   - Check contract is actively traded
   - Ensure proper exchange permissions

3. **Order Rejected**
   ```
   Trade rejected by risk management
   ```
   - Check account balance
   - Verify position limits
   - Review daily loss limits
   - Check margin requirements

4. **No Market Data**
   ```
   No live price available
   ```
   - Verify market data subscriptions
   - Check market hours
   - Ensure real-time data permissions

### Debug Mode

Enable debug logging:

```python
import structlog
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(structlog.DEBUG)
)
```

### Support

For IB-specific issues:
- IB API Documentation: https://interactivebrokers.github.io/
- IB Client Portal: https://www.interactivebrokers.com/
- ib_insync Documentation: https://ib-insync.readthedocs.io/

## Security Considerations

1. **API Access**
   - Limit connections to localhost only
   - Use strong passwords for IB accounts
   - Enable two-factor authentication

2. **Environment Variables**
   - Never commit `.env` files to version control
   - Use secure key management in production
   - Rotate API keys regularly

3. **Live Trading**
   - Start with paper trading
   - Use position limits
   - Monitor trades actively
   - Have emergency stop procedures

## License

This IB integration is provided under the MIT License. Interactive Brokers and TWS are trademarks of Interactive Brokers LLC.