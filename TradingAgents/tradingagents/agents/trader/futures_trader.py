import functools


def create_futures_trader(llm, memory):
    def futures_trader_node(state, name):
        futures_symbol = state["futures_symbol"]
        investment_plan = state["investment_plan"]
        futures_market_report = state.get("futures_market_report", "")
        sentiment_report = state.get("sentiment_report", "")
        news_report = state.get("news_report", "")
        futures_fundamentals_report = state.get("futures_fundamentals_report", "")

        curr_situation = f"{futures_market_report}\n\n{sentiment_report}\n\n{news_report}\n\n{futures_fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for rec in past_memories:
            past_memory_str += rec["recommendation"] + "\n\n"

        context = {
            "role": "user",
            "content": f"Based on comprehensive analysis by specialized futures analysts, here is an investment plan for {futures_symbol}. This plan incorporates futures-specific insights including market structure analysis, technical indicators optimized for futures trading, macroeconomic factors, and commodity-specific fundamentals. Use this plan as a foundation for your futures trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nLeverage these insights to make an informed futures trading decision, considering the unique characteristics of futures contracts including leverage, margin requirements, and expiry considerations.",
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are a specialized futures trading agent with expertise in commodity and financial futures markets. Your role is to analyze market data and make informed trading decisions for futures contracts.

**Key Futures Trading Considerations:**

**Leverage & Margin:**
- Futures contracts are highly leveraged instruments
- Consider margin requirements and potential margin calls
- Account for daily mark-to-market settlements
- Factor in maintenance margin levels

**Contract Specifications:**
- Understand contract size and tick value
- Monitor expiry dates and rollover requirements
- Consider first notice dates for physical delivery contracts
- Account for contract specifications when sizing positions

**Market Structure:**
- Analyze contango/backwardation implications
- Consider basis relationships between futures and spot
- Evaluate roll yield potential
- Monitor curve structure changes

**Risk Management:**
- Implement strict position sizing due to leverage
- Use stop-losses based on futures-specific volatility (ATR multiples)
- Consider time decay effects near expiry
- Monitor overnight gaps and limit moves

**Seasonal & Cyclical Factors:**
- Account for seasonal patterns in commodity futures
- Consider storage costs and convenience yield
- Factor in harvest/production cycles
- Monitor weather impacts for agricultural futures

**Volume & Liquidity:**
- Ensure adequate volume for entry/exit
- Consider bid-ask spreads and market impact
- Monitor open interest trends
- Avoid illiquid contract months

**Decision Framework:**
1. Assess fundamental outlook for underlying asset
2. Evaluate technical setup and trend strength
3. Consider futures-specific factors (basis, roll yield, expiry)
4. Determine position size based on margin and volatility
5. Set stop-loss levels using futures-appropriate methods
6. Plan rollover strategy if holding through expiry

Based on your analysis, provide a specific recommendation to BUY, SELL, or HOLD the futures contract.
Always conclude with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' along with:
- Recommended position size
- Stop-loss level
- Target price (if applicable)
- Rollover considerations

Learn from past similar situations: {past_memory_str}""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "futures_trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(futures_trader_node, name="FuturesTrader")