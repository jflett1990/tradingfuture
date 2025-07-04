from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json


def create_futures_fundamentals_analyst(llm, toolkit):
    def futures_fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        symbol = state["futures_symbol"]
        
        if toolkit.config["online_tools"]:
            tools = [
                toolkit.get_futures_info,
                toolkit.get_contango_backwardation_analysis,
                toolkit.get_futures_fundamentals,
                toolkit.get_volume_open_interest,
                toolkit.get_expiry_analysis,
                toolkit.get_margin_requirements,
            ]
        else:
            tools = [
                toolkit.get_futures_info,
                toolkit.get_futures_fundamentals,
                toolkit.get_margin_requirements,
            ]

        system_message = (
            "You are a futures market analyst specialized in fundamental analysis of commodity and financial futures. "
            "Your task is to analyze the fundamental factors affecting the futures contract including: "
            "1. Market structure analysis (contango/backwardation) "
            "2. Supply and demand fundamentals specific to the underlying commodity/asset "
            "3. Storage costs, convenience yield, and carrying costs "
            "4. Contract specifications (expiry, margin requirements, contract size) "
            "5. Seasonal patterns and cyclical factors "
            "6. Volume and open interest trends "
            "7. Basis relationships between futures and spot prices "
            "8. Roll yield considerations "
            "9. Geopolitical and macroeconomic factors affecting the underlying "
            "10. Weather patterns (for agricultural commodities) or economic indicators (for financial futures) "
            "\n\nFor commodity futures, focus on: "
            "- Production and consumption data "
            "- Inventory levels and storage capacity "
            "- Export/import dynamics "
            "- Weather and seasonal factors "
            "- Substitute products and demand elasticity "
            "\n\nFor financial futures, focus on: "
            "- Economic indicators and central bank policies "
            "- Interest rate environment "
            "- Currency relationships "
            "- Market volatility and risk sentiment "
            "- Correlation with underlying cash markets "
            "\n\nProvide detailed analysis with specific data points and avoid generic statements. "
            "Include a risk assessment for holding positions through contract expiry. "
            "Make sure to append a Markdown table at the end of the report organizing key fundamentals."
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
            "futures_fundamentals_report": report,
        }

    return futures_fundamentals_analyst_node