"""analyst_nodes.py – Prompt‑factory utilities for LangChain analyst agents.

This module exports two helper factories:

• `create_futures_market_analyst` – builds a technical‑analysis agent.
• `create_futures_fundamentals_analyst` – builds a fundamentals agent.

Both return **node callables** that fit seamlessly into LangGraph / LangChain state
machines (`state -> dict`). The design removes code duplication, fixes minor bugs
(trailing dot, missing tool calls check), and supports both *offline* and
*online* tool‑sets via `toolkit.config["online_tools"]`.

Each analyst:
  • Injects a domain‑rich `system_message` detailing its analysis mandate.
  • Automatically populates `{tool_names}`, `{current_date}`, and `{symbol}` in
    the prompt via `partial()`.
  • Enforces a first‑step call to `get_futures_data` by placing it at index 0 of
    the tools list – LangChain's planner picks left‑most tools first.
  • Returns `{"messages": [...], "<report_key>": str}` where `<report_key>` is
    either `futures_market_report` or `futures_fundamentals_report`.

Usage example
-------------
```python
llm = ChatOpenAI(model="gpt-4o" , temperature=0)
market_node = create_futures_market_analyst(llm, my_toolkit)
state = {"trade_date": "2025-07-03", "futures_symbol": "CL", "messages": []}
result = asyncio.run(market_node(state))
print(result["futures_market_report"])
```
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel

# ---------------------------------------------------------------------------
#  INTERNAL HELPER
# ---------------------------------------------------------------------------

def _make_node(
    *,
    llm: BaseChatModel,
    toolkit,
    report_key: str,
    system_message: str,
    online_tools: Sequence,
    offline_tools: Sequence,
) -> Callable[[Dict], Dict]:
    """Return a state‑transform function conforming to LangGraph node API."""

    def _node(state: Dict) -> Dict:  # noqa: D401
        current_date: str = state["trade_date"]
        symbol: str = state["futures_symbol"]
        tools = list(online_tools if toolkit.config.get("online_tools", False) else offline_tools)

        # Ensure get_futures_data is always first for planner prioritisation.
        tools.sort(key=lambda t: 0 if t.__name__ == "get_futures_data" else 1)
        tool_names = ", ".join(t.name for t in tools)

        sys_prompt = (
            "You are a helpful AI assistant, collaborating with other assistants. "
            "Use the provided tools to progress towards answering the question. "
            "If you are unable to fully answer, that's OK; another assistant will continue. "
            "If you generate a FINAL TRANSACTION PROPOSAL (BUY/HOLD/SELL) or final deliverable, "
            "prefix with 'FINAL TRANSACTION PROPOSAL:' so the team knows to stop. "
            "You have access to the following tools: {tool_names} \n{system_message}\n\n"
            "Current date: {current_date}. Futures contract under review: {symbol}."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", sys_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        ).partial(
            system_message=system_message,
            tool_names=tool_names,
            current_date=current_date,
            symbol=symbol,
        )

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])
        report = "" if getattr(result, "tool_calls", []) else result.content

        return {"messages": [result], report_key: report}

    return _node

# ---------------------------------------------------------------------------
#  PUBLIC FACTORIES
# ---------------------------------------------------------------------------

def create_futures_market_analyst(llm: BaseChatModel, toolkit) -> Callable[[Dict], Dict]:  # noqa: D401
    """Factory for a *technical* futures analyst node."""

    system_message = (
        "You are a futures market *technical* analyst specialised in commodity and "
        "financial futures. Select ≤8 complementary indicators (avoid redundancy) "
        "from Moving Averages, MACD family, Momentum oscillators, Volatility, Volume, "
        "plus futures‑specific factors. Provide granular trend/volatility/volume "
        "analysis, contract‑specific nuances (expiry, rollover), S/R levels, and risk "
        "management insights. Conclude with a Markdown table of key technical levels. "
        "Always call `get_futures_data` first."
    )

    online = [
        toolkit.get_futures_data,
        toolkit.get_futures_technical_indicators,
        toolkit.get_futures_momentum_analysis,
        toolkit.get_futures_volatility_analysis,
    ]
    offline = [toolkit.get_futures_data, toolkit.get_futures_technical_indicators]

    return _make_node(
        llm=llm,
        toolkit=toolkit,
        report_key="futures_market_report",
        system_message=system_message,
        online_tools=online,
        offline_tools=offline,
    )


def create_futures_fundamentals_analyst(llm: BaseChatModel, toolkit) -> Callable[[Dict], Dict]:  # noqa: D401
    """Factory for a *fundamental* futures analyst node."""

    system_message = (
        "You are a futures *fundamental* analyst. Analyse market structure "
        "(contango/backwardation), supply‑demand, inventory, roll yield, margin, "
        "volume & OI, geopolitical & macro factors, plus contract specs. Avoid "
        "generic statements; cite concrete data. Finish with a Markdown table of "
        "key fundamentals."
    )

    online = [
        toolkit.get_futures_info,
        toolkit.get_contango_backwardation_analysis,
        toolkit.get_futures_fundamentals,
        toolkit.get_volume_open_interest,
        toolkit.get_expiry_analysis,
        toolkit.get_margin_requirements,
    ]
    offline = [
        toolkit.get_futures_info,
        toolkit.get_futures_fundamentals,
        toolkit.get_margin_requirements,
    ]

    return _make_node(
        llm=llm,
        toolkit=toolkit,
        report_key="futures_fundamentals_report",
        system_message=system_message,
        online_tools=online,
        offline_tools=offline,
    )