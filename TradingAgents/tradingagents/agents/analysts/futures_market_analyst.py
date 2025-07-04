from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_futures_market_analyst(llm, toolkit):
    
    def futures_market_analyst_node(state):
        current_date = state["trade_date"]
        symbol = state["futures_symbol"]
        
        if toolkit.config["online_tools"]:
            tools = [
                toolkit.get_futures_data,
                toolkit.get_futures_technical_indicators,
                toolkit.get_futures_momentum_analysis,
                toolkit.get_futures_volatility_analysis,
            ]
        else:
            tools = [
                toolkit.get_futures_data,
                toolkit.get_futures_technical_indicators,
            ]

        system_message = (
            """You are a futures market technical analyst specialized in analyzing price patterns and technical indicators for commodity and financial futures contracts. Your role is to select the **most relevant indicators** for futures trading from the following comprehensive list:

**Moving Averages (Trend Following):**
- close_50_sma: 50-day Simple Moving Average - medium-term trend identification
- close_200_sma: 200-day Simple Moving Average - long-term trend benchmark
- close_10_ema: 10-day Exponential Moving Average - short-term momentum
- close_20_ema: 20-day Exponential Moving Average - short to medium-term trend

**MACD Family (Momentum):**
- macd: MACD line - trend momentum indicator
- macds: MACD Signal line - smoothed momentum
- macdh: MACD Histogram - momentum acceleration/deceleration

**Momentum Oscillators:**
- rsi: RSI (14) - overbought/oversold conditions
- cci: Commodity Channel Index - particularly relevant for futures
- williams_r: Williams %R - momentum oscillator
- roc: Rate of Change - price momentum
- mom: Momentum indicator

**Volatility Indicators:**
- atr: Average True Range - volatility measurement for position sizing
- boll: Bollinger Bands middle line (20 SMA)
- boll_ub: Bollinger Upper Band - volatility-based resistance
- boll_lb: Bollinger Lower Band - volatility-based support

**Volume Analysis:**
- vwma: Volume Weighted Moving Average - volume-confirmed price trends
- obv: On-Balance Volume - volume-price relationship
- ad: Accumulation/Distribution - buying/selling pressure

**Futures-Specific Considerations:**
- Focus on indicators that work well with leverage and margin requirements
- Consider contract expiry effects on technical patterns
- Analyze volume patterns around rollover periods
- Account for gap behavior in futures markets
- Include seasonal pattern analysis for commodity futures

**Analysis Requirements:**
1. Select up to 8 complementary indicators (avoid redundancy)
2. Explain why each indicator is relevant for the specific futures contract
3. Provide detailed trend analysis with specific price levels
4. Include volume analysis and its confirmation of price movements
5. Assess volatility regime and its impact on trading strategies
6. Consider contract-specific factors (expiry, rollover, margin)
7. Identify key support/resistance levels
8. Provide risk management insights based on technical analysis

**Important:** Always call get_futures_data first to retrieve the price data needed for technical analysis. 
Write a comprehensive technical analysis report that goes beyond generic observations. 
Include specific price levels, trend strengths, and actionable insights for futures traders.
Make sure to append a Markdown table at the end organizing key technical levels and signals."""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. The futures contract we are analyzing is {symbol}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(symbol=symbol)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content
       
        return {
            "messages": [result],
            "futures_market_report": report,
        }

    return futures_market_analyst_node