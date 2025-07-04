import functools
import json
import re


def create_futures_risk_manager(llm, memory):
    def futures_risk_manager_node(state, name):
        futures_symbol = state["futures_symbol"]
        trader_recommendation = state.get("futures_trader_investment_plan", "")
        market_analysis = state.get("futures_market_report", "")
        fundamentals_analysis = state.get("futures_fundamentals_report", "")
        
        # Extract trading action from trader recommendation
        action_match = re.search(r'FINAL TRANSACTION PROPOSAL:\s*\*\*(BUY|SELL|HOLD)\*\*', trader_recommendation)
        proposed_action = action_match.group(1) if action_match else "UNKNOWN"
        
        context = {
            "role": "user",
            "content": f"The futures trader has proposed a {proposed_action} recommendation for {futures_symbol}. Please conduct a comprehensive risk assessment of this proposal considering futures-specific risk factors.\n\nTrader's Recommendation:\n{trader_recommendation}\n\nMarket Analysis:\n{market_analysis}\n\nFundamentals Analysis:\n{fundamentals_analysis}",
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are a specialized futures risk management expert responsible for evaluating trading proposals for futures contracts. Your role is to conduct comprehensive risk assessments that account for the unique characteristics of futures trading.

**Futures-Specific Risk Assessment Framework:**

**1. Leverage & Margin Risks:**
- Evaluate margin requirements vs. account size
- Assess potential for margin calls during adverse moves
- Consider daily mark-to-market impact on cash flow
- Calculate maximum potential daily loss
- Review maintenance margin buffer

**2. Contract-Specific Risks:**
- Analyze time to expiry and rollover requirements
- Assess liquidity risks, especially in distant months
- Consider first notice date implications for physical delivery
- Evaluate contract size appropriateness for portfolio
- Review tick size and minimum price movement impact

**3. Market Structure Risks:**
- Evaluate contango/backwardation implications
- Assess basis risk between futures and underlying
- Consider roll yield impact on returns
- Monitor curve structure stability
- Evaluate spread relationships

**4. Volatility & Gap Risks:**
- Assess futures-specific volatility patterns
- Consider overnight gap potential
- Evaluate limit move scenarios
- Review historical volatility vs. implied volatility
- Consider volatility clustering effects

**5. Operational Risks:**
- Assess trading hours and after-hours exposure
- Consider settlement procedures and timing
- Evaluate technology/system risks
- Review counterparty risks (exchange/clearing)
- Consider regulatory compliance requirements

**6. Portfolio & Concentration Risks:**
- Evaluate position size relative to total portfolio
- Assess correlation with existing positions
- Consider sector/commodity concentration
- Review geographical/political exposure
- Evaluate currency risks for international futures

**7. Seasonal & Cyclical Risks:**
- Consider seasonal volatility patterns
- Assess harvest/production cycle impacts
- Review weather-related risks
- Consider geopolitical event timing
- Evaluate macroeconomic cycle position

**Risk Mitigation Strategies:**
- Position sizing recommendations
- Stop-loss placement guidelines
- Hedging strategies
- Diversification requirements
- Risk monitoring protocols

**Decision Framework:**
1. Quantify maximum potential loss scenarios
2. Assess probability of margin call
3. Evaluate impact on overall portfolio risk
4. Consider risk-adjusted return potential
5. Review compliance with risk limits
6. Provide specific risk management recommendations

Your assessment should result in one of these recommendations:
- APPROVE: Low to moderate risk, acceptable for execution
- APPROVE WITH CONDITIONS: Moderate risk, requires specific risk controls
- MODIFY: High risk, requires position size reduction or other modifications
- REJECT: Unacceptable risk level

Always conclude with 'FINAL RISK ASSESSMENT: **APPROVE/APPROVE WITH CONDITIONS/MODIFY/REJECT**' and provide specific risk management requirements.""",
            },
            context,
        ]

        result = llm.invoke(messages)

        # Extract risk assessment decision
        risk_decision_match = re.search(r'FINAL RISK ASSESSMENT:\s*\*\*(APPROVE|APPROVE WITH CONDITIONS|MODIFY|REJECT)\*\*', result.content)
        risk_decision = risk_decision_match.group(1) if risk_decision_match else "UNKNOWN"

        return {
            "messages": [result],
            "futures_risk_assessment": result.content,
            "risk_decision": risk_decision,
            "sender": name,
        }

    return functools.partial(futures_risk_manager_node, name="FuturesRiskManager")