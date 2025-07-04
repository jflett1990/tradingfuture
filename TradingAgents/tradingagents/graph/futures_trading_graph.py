import operator
from typing import Annotated, Sequence, TypedDict
import functools

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage

from ..agents.analysts.futures_fundamentals_analyst import create_futures_fundamentals_analyst
from ..agents.analysts.futures_market_analyst import create_futures_market_analyst
from ..agents.analysts.news_analyst import create_news_analyst
from ..agents.analysts.social_media_analyst import create_social_media_analyst
from ..agents.researchers.bull_researcher import create_bull_researcher
from ..agents.researchers.bear_researcher import create_bear_researcher
from ..agents.managers.research_manager import create_research_manager
from ..agents.trader.futures_trader import create_futures_trader
from ..agents.risk_mgmt.futures_risk_manager import create_futures_risk_manager
from ..agents.utils.memory import Memory
from ..dataflows.interface import DataFlowToolkit


class FuturesTradingState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    futures_symbol: str
    trade_date: str
    sender: str
    
    # Analysis reports
    futures_market_report: str
    futures_fundamentals_report: str
    sentiment_report: str
    news_report: str
    
    # Research outputs
    bull_research: str
    bear_research: str
    investment_plan: str
    
    # Trading decisions
    futures_trader_investment_plan: str
    futures_risk_assessment: str
    risk_decision: str
    
    # Final decision
    final_decision: str


class FuturesTradingGraph:
    def __init__(self, debug: bool = False, config: dict = None):
        self.debug = debug
        self.config = config or {}
        self.memory = Memory()
        self.toolkit = DataFlowToolkit(config=self.config)
        
        # Initialize LLMs
        from langchain_openai import ChatOpenAI
        self.deep_think_llm = ChatOpenAI(
            model=self.config.get("deep_think_llm", "gpt-4o-mini"),
            temperature=0.1,
            api_key=self.config.get("openai_api_key")
        )
        self.quick_think_llm = ChatOpenAI(
            model=self.config.get("quick_think_llm", "gpt-4o-mini"),
            temperature=0.1,
            api_key=self.config.get("openai_api_key")
        )
        
        # Create agent nodes
        self.futures_fundamentals_analyst = create_futures_fundamentals_analyst(
            self.deep_think_llm, self.toolkit
        )
        self.futures_market_analyst = create_futures_market_analyst(
            self.deep_think_llm, self.toolkit
        )
        self.news_analyst = create_news_analyst(
            self.quick_think_llm, self.toolkit
        )
        self.social_media_analyst = create_social_media_analyst(
            self.quick_think_llm, self.toolkit
        )
        self.bull_researcher = create_bull_researcher(
            self.deep_think_llm, self.memory
        )
        self.bear_researcher = create_bear_researcher(
            self.deep_think_llm, self.memory
        )
        self.research_manager = create_research_manager(
            self.deep_think_llm, self.memory
        )
        self.futures_trader = create_futures_trader(
            self.deep_think_llm, self.memory
        )
        self.futures_risk_manager = create_futures_risk_manager(
            self.deep_think_llm, self.memory
        )
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(FuturesTradingState)
        
        # Add nodes
        workflow.add_node("futures_fundamentals_analyst", self.futures_fundamentals_analyst)
        workflow.add_node("futures_market_analyst", self.futures_market_analyst)
        workflow.add_node("news_analyst", self.news_analyst)
        workflow.add_node("social_media_analyst", self.social_media_analyst)
        workflow.add_node("bull_researcher", self.bull_researcher)
        workflow.add_node("bear_researcher", self.bear_researcher)
        workflow.add_node("research_manager", self.research_manager)
        workflow.add_node("futures_trader", self.futures_trader)
        workflow.add_node("futures_risk_manager", self.futures_risk_manager)
        
        # Define the flow
        workflow.set_entry_point("futures_fundamentals_analyst")
        
        # Parallel analysis phase
        workflow.add_edge("futures_fundamentals_analyst", "futures_market_analyst")
        workflow.add_edge("futures_market_analyst", "news_analyst")
        workflow.add_edge("news_analyst", "social_media_analyst")
        
        # Research phase
        workflow.add_edge("social_media_analyst", "bull_researcher")
        workflow.add_edge("bull_researcher", "bear_researcher")
        workflow.add_edge("bear_researcher", "research_manager")
        
        # Trading decision phase
        workflow.add_edge("research_manager", "futures_trader")
        workflow.add_edge("futures_trader", "futures_risk_manager")
        
        # Risk management decision
        workflow.add_conditional_edges(
            "futures_risk_manager",
            self._risk_decision_router,
            {
                "approved": END,
                "rejected": END,
                "modified": END,
            }
        )
        
        return workflow.compile()
    
    def _risk_decision_router(self, state: FuturesTradingState) -> str:
        """Route based on risk management decision"""
        risk_decision = state.get("risk_decision", "UNKNOWN")
        
        if risk_decision in ["APPROVE", "APPROVE WITH CONDITIONS"]:
            return "approved"
        elif risk_decision == "REJECT":
            return "rejected"
        elif risk_decision == "MODIFY":
            return "modified"
        else:
            return "approved"  # Default case
    
    def propagate(self, symbol: str, trade_date: str):
        """Execute futures trading analysis for given symbol and date"""
        
        # Ensure symbol is in futures format
        if not symbol.endswith('=F'):
            symbol = symbol + '=F'
        
        initial_state = FuturesTradingState(
            messages=[],
            futures_symbol=symbol,
            trade_date=trade_date,
            sender="system",
            futures_market_report="",
            futures_fundamentals_report="",
            sentiment_report="",
            news_report="",
            bull_research="",
            bear_research="",
            investment_plan="",
            futures_trader_investment_plan="",
            futures_risk_assessment="",
            risk_decision="",
            final_decision=""
        )
        
        if self.debug:
            print(f"Starting futures trading analysis for {symbol} on {trade_date}")
        
        # Execute the graph
        final_state = self.graph.invoke(initial_state)
        
        # Extract final decision
        trader_decision = final_state.get("futures_trader_investment_plan", "")
        risk_assessment = final_state.get("futures_risk_assessment", "")
        risk_decision = final_state.get("risk_decision", "UNKNOWN")
        
        final_decision = {
            "symbol": symbol,
            "date": trade_date,
            "trader_recommendation": trader_decision,
            "risk_assessment": risk_assessment,
            "risk_decision": risk_decision,
            "analysis_reports": {
                "fundamentals": final_state.get("futures_fundamentals_report", ""),
                "technical": final_state.get("futures_market_report", ""),
                "sentiment": final_state.get("sentiment_report", ""),
                "news": final_state.get("news_report", ""),
                "research": final_state.get("investment_plan", "")
            }
        }
        
        if self.debug:
            print(f"Futures trading analysis completed for {symbol}")
            print(f"Risk Decision: {risk_decision}")
        
        return final_state, final_decision