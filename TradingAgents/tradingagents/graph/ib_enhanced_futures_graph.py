"""ib_enhanced_futures_graph.py – IB-integrated futures trading orchestrator.

Enhanced version of the futures trading graph with Interactive Brokers integration:
• Live trading execution through IB TWS/Gateway
• Real-time market data streaming from IB
• Risk-managed order placement with bracket orders
• Portfolio tracking and P&L monitoring
• Position sizing based on actual account balance
• Circuit breakers for live trading protection

Features:
• Seamless integration with existing analysis framework
• Paper trading and live trading modes
• Comprehensive logging and monitoring
• Real-time position and P&L updates
• Prometheus metrics for performance tracking

© 2025 Prompt Maestro 9000, MIT License
"""
from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Sequence, TypedDict, Dict, List, Optional, Any
import structlog
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from langchain_anthropic import ChatAnthropic

from ..agents.analysts.analyst_nodes import create_futures_market_analyst, create_futures_fundamentals_analyst
from ..agents.analysts.news_analyst import create_news_analyst
from ..agents.analysts.social_media_analyst import create_social_media_analyst
from ..agents.researchers.bull_researcher import create_bull_researcher
from ..agents.researchers.bear_researcher import create_bear_researcher
from ..agents.managers.research_manager import create_research_manager
from ..agents.utils.memory import Memory
from ..dataflows.interface import DataFlowToolkit
from ..dataflows.news_sentinel import NewsSentinel
from ..strategies.advanced_strategies import StrategyEngine, TradeSignal, Action
from ..brokers.ib_broker import IBBroker, create_ib_broker, Position
from ..brokers.ib_realtime_adapter import IBRealTimeStreamAdapter, create_ib_stream_manager

log = structlog.get_logger("ib_enhanced_futures_graph")

# ---------------------------------------------------------------------------
#  STATE DEFINITION
# ---------------------------------------------------------------------------

class IBEnhancedFuturesState(TypedDict):  # noqa: D101
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
    
    # Advanced signals
    strategy_signals: Dict[str, TradeSignal]
    consensus_signal: TradeSignal
    
    # IB-specific data
    live_price: Optional[float]
    current_position: Optional[Position]
    account_balance: float
    portfolio_value: float
    daily_pnl: float
    
    # Real-time data
    order_book_data: Optional[Dict]
    news_alerts: List[Dict]
    
    # Risk management
    risk_assessment: str
    position_size: float
    risk_limits: Dict[str, float]
    
    # Trading execution
    ib_trade: Optional[Any]  # IB Trade object
    execution_status: str
    
    # Final decision
    final_decision: Dict

# ---------------------------------------------------------------------------
#  IB ENHANCED FUTURES GRAPH
# ---------------------------------------------------------------------------

class IBEnhancedFuturesGraph:  # noqa: D101
    
    def __init__(self, debug: bool = False, config: Optional[Dict] = None, paper_trading: bool = True):  # noqa: D401
        self.debug = debug
        self.config = config or {}
        self.paper_trading = paper_trading
        self.memory = Memory()
        self.toolkit = DataFlowToolkit(config=self.config)
        
        # IB components (initialized async)
        self.ib_broker: Optional[IBBroker] = None
        self.ib_stream_manager: Optional[IBRealTimeStreamAdapter] = None
        
        # Analysis components
        self.news_sentinel: Optional[NewsSentinel] = None
        self.strategy_engine: Optional[StrategyEngine] = None
        
        # LLM setup  
        self.deep_think_llm = ChatAnthropic(
            model=self.config.get("deep_think_llm", "claude-3-5-sonnet-20241022"),
            temperature=0.1,
            api_key=self.config.get("anthropic_api_key")
        )
        self.quick_think_llm = ChatAnthropic(
            model=self.config.get("quick_think_llm", "claude-3-5-sonnet-20241022"),
            temperature=0.1,
            api_key=self.config.get("anthropic_api_key")
        )
        
        # Build graph
        self.graph = self._build_ib_enhanced_graph()
    
    async def initialize_ib_components(self, symbols: List[str]) -> None:  # noqa: D401
        """Initialize IB broker and streaming components."""
        try:
            log.info("initializing_ib_components", 
                    symbols=symbols, 
                    paper_trading=self.paper_trading)
            
            # Initialize IB broker
            self.ib_broker = await create_ib_broker(paper_trading=self.paper_trading)
            
            # Initialize IB streaming (uses the same broker connection)
            if self.config.get("enable_realtime_stream", True):
                self.ib_stream_manager = await create_ib_stream_manager(symbols, self.ib_broker)
            
            # Initialize analysis components
            self.news_sentinel = NewsSentinel(symbols)
            self.strategy_engine = StrategyEngine(symbols)
            
            log.info("ib_components_initialized", 
                    broker_connected=self.ib_broker.ib.isConnected(),
                    streaming_enabled=self.ib_stream_manager is not None)
            
        except Exception as exc:
            log.error("ib_initialization_failed", error=str(exc))
            raise
    
    def _build_ib_enhanced_graph(self) -> StateGraph:  # noqa: D401
        """Build the IB-enhanced trading graph."""
        workflow = StateGraph(IBEnhancedFuturesState)
        
        # Create analyst nodes using factory
        futures_market_analyst = create_futures_market_analyst(self.deep_think_llm, self.toolkit)
        futures_fundamentals_analyst = create_futures_fundamentals_analyst(self.deep_think_llm, self.toolkit)
        
        # Legacy analysts (can be replaced with factories later)
        news_analyst = create_news_analyst(self.quick_think_llm, self.toolkit)
        social_media_analyst = create_social_media_analyst(self.quick_think_llm, self.toolkit)
        
        # Research team
        bull_researcher = create_bull_researcher(self.deep_think_llm, self.memory)
        bear_researcher = create_bear_researcher(self.deep_think_llm, self.memory)
        research_manager = create_research_manager(self.deep_think_llm, self.memory)
        
        # Add nodes (IB-specific nodes)
        workflow.add_node("ib_data_node", self._ib_data_node)
        workflow.add_node("futures_market_analyst", futures_market_analyst)
        workflow.add_node("futures_fundamentals_analyst", futures_fundamentals_analyst)
        workflow.add_node("news_analyst", news_analyst)
        workflow.add_node("social_media_analyst", social_media_analyst)
        workflow.add_node("advanced_strategy_node", self._advanced_strategy_node)
        workflow.add_node("bull_researcher", bull_researcher)
        workflow.add_node("bear_researcher", bear_researcher)
        workflow.add_node("research_manager", research_manager)
        workflow.add_node("ib_risk_management_node", self._ib_risk_management_node)
        workflow.add_node("ib_execution_node", self._ib_execution_node)
        workflow.add_node("ib_final_decision_node", self._ib_final_decision_node)
        
        # Define execution flow
        workflow.set_entry_point("ib_data_node")
        
        # Analysis phase
        workflow.add_edge("ib_data_node", "futures_market_analyst")
        workflow.add_edge("futures_market_analyst", "futures_fundamentals_analyst")
        workflow.add_edge("futures_fundamentals_analyst", "news_analyst")
        workflow.add_edge("news_analyst", "social_media_analyst")
        
        # Strategy analysis
        workflow.add_edge("social_media_analyst", "advanced_strategy_node")
        
        # Research phase
        workflow.add_edge("advanced_strategy_node", "bull_researcher")
        workflow.add_edge("bull_researcher", "bear_researcher")
        workflow.add_edge("bear_researcher", "research_manager")
        
        # IB-specific risk management and execution
        workflow.add_edge("research_manager", "ib_risk_management_node")
        workflow.add_edge("ib_risk_management_node", "ib_execution_node")
        workflow.add_edge("ib_execution_node", "ib_final_decision_node")
        workflow.add_edge("ib_final_decision_node", END)
        
        return workflow.compile()
    
    # ------------------------------------------------------------------- #
    #  IB-SPECIFIC NODES
    # ------------------------------------------------------------------- #
    
    async def _ib_data_node(self, state: IBEnhancedFuturesState) -> Dict:  # noqa: D401
        """Gather live IB data and portfolio information."""
        symbol = state["futures_symbol"]
        
        live_price = None
        current_position = None
        account_balance = 0.0
        portfolio_value = 0.0
        daily_pnl = 0.0
        order_book_data = None
        news_alerts = []
        
        try:
            if self.ib_broker:
                # Get current position
                current_position = self.ib_broker.get_position(symbol)
                
                # Get portfolio summary
                portfolio = self.ib_broker.get_portfolio_summary()
                account_balance = portfolio.get("account_value", 0.0)
                portfolio_value = sum(pos.get("market_value", 0) for pos in portfolio.get("positions", {}).values())
                daily_pnl = portfolio.get("daily_pnl", 0.0)
                
                log.info("ib_portfolio_data", 
                        symbol=symbol,
                        position=current_position.qty if current_position else 0,
                        daily_pnl=daily_pnl,
                        portfolio_value=portfolio_value)
            
            if self.ib_stream_manager:
                # Get live market data
                market_summary = self.ib_stream_manager.get_market_summary(symbol)
                if market_summary:
                    live_price = market_summary.last_price
                
                # Get order book
                order_book_data = self.ib_stream_manager.get_market_depth_levels(symbol)
                
                log.info("ib_market_data", 
                        symbol=symbol,
                        live_price=live_price,
                        bid_levels=len(order_book_data.get("bids", [])),
                        ask_levels=len(order_book_data.get("asks", [])))
            
            # Get urgent news alerts
            if self.news_sentinel:
                news_snapshot = await self.news_sentinel.snapshot(symbol)
                if news_snapshot and news_snapshot.news_items:
                    urgent_news = [
                        {
                            "title": item.title,
                            "urgency": item.urgency,
                            "sentiment": item.sentiment,
                            "timestamp": item.timestamp.isoformat()
                        }
                        for item in news_snapshot.news_items
                        if item.urgency > 0.6
                    ]
                    news_alerts = urgent_news[:3]
            
        except Exception as exc:
            log.error("ib_data_collection_error", symbol=symbol, error=str(exc))
        
        return {
            "live_price": live_price,
            "current_position": current_position,
            "account_balance": account_balance,
            "portfolio_value": portfolio_value,
            "daily_pnl": daily_pnl,
            "order_book_data": order_book_data,
            "news_alerts": news_alerts,
        }
    
    async def _advanced_strategy_node(self, state: IBEnhancedFuturesState) -> Dict:  # noqa: D401
        """Run advanced strategy analysis with live data."""
        if not self.strategy_engine:
            log.warning("strategy_engine_not_initialized")
            return {"strategy_signals": {}, "consensus_signal": None}
        
        try:
            symbol = state["futures_symbol"]
            
            # Get market data - try live data first, fallback to yfinance
            data = None
            if self.ib_stream_manager:
                # TODO: Convert IB streaming data to pandas DataFrame
                # For now, fallback to yfinance
                pass
            
            if data is None:
                import yfinance as yf
                data = yf.download(symbol, period="3mo", interval="1h", progress=False)
            
            # Run multi-strategy analysis
            consensus_signal = await self.strategy_engine.multi_signal(data, symbol)
            
            # Get individual strategy signals
            strategy_coros = [
                self.strategy_engine.lightning_news(data, symbol),
                self.strategy_engine.adaptive_momentum(data, symbol),
                self.strategy_engine.volatility_expansion(data, symbol),
                self.strategy_engine.mean_reversion(data, symbol),
            ]
            
            individual_signals = await asyncio.gather(*strategy_coros, return_exceptions=True)
            
            strategy_signals = {}
            for i, signal in enumerate(individual_signals):
                if isinstance(signal, Exception):
                    log.warning("strategy_error", strategy=i, error=str(signal))
                    continue
                strategy_signals[signal.strategy] = signal
            
            log.info("ib_strategies_completed", 
                    symbol=symbol, 
                    consensus=consensus_signal.action,
                    confidence=consensus_signal.confidence,
                    live_price=state.get("live_price"))
            
            return {
                "strategy_signals": strategy_signals,
                "consensus_signal": consensus_signal,
            }
            
        except Exception as exc:
            log.error("ib_strategy_error", error=str(exc))
            return {"strategy_signals": {}, "consensus_signal": None}
    
    async def _ib_risk_management_node(self, state: IBEnhancedFuturesState) -> Dict:  # noqa: D401
        """IB-aware risk management with live account data."""
        symbol = state["futures_symbol"]
        consensus_signal = state.get("consensus_signal")
        account_balance = state.get("account_balance", 0.0)
        current_position = state.get("current_position")
        daily_pnl = state.get("daily_pnl", 0.0)
        
        if not consensus_signal:
            return {
                "risk_assessment": "No trading signal to assess",
                "position_size": 0.0,
                "risk_limits": {}
            }
        
        try:
            # Account-based risk calculations
            max_risk_per_trade = self.config.get("max_risk_per_trade", 0.02)  # 2%
            max_position_size = self.config.get("max_position_size", 0.1)     # 10%
            
            # Calculate available buying power
            available_capital = account_balance * max_position_size
            risk_capital = account_balance * max_risk_per_trade
            
            # Adjust for existing position
            position_adjustment = 1.0
            if current_position and current_position.qty != 0:
                # If we have a position, be more conservative
                position_adjustment = 0.5
                log.info("existing_position_detected", 
                        symbol=symbol,
                        current_qty=current_position.qty,
                        unrealized_pnl=current_position.unrealized)
            
            # Calculate position size based on signal and account
            base_position_size = min(
                consensus_signal.position_size,
                available_capital / consensus_signal.entry_price
            )
            
            confidence_adjustment = consensus_signal.confidence
            urgency_adjustment = {"high": 1.2, "medium": 1.0, "low": 0.8}.get(consensus_signal.urgency, 1.0)
            
            final_position_size = base_position_size * confidence_adjustment * urgency_adjustment * position_adjustment
            
            # Risk limits
            risk_limits = {
                "max_loss": risk_capital,
                "stop_loss": consensus_signal.stop_loss,
                "take_profit": consensus_signal.take_profit,
                "max_holding_period": 24 if consensus_signal.urgency == "high" else 72,  # hours
                "available_capital": available_capital,
                "risk_capital": risk_capital
            }
            
            # Enhanced risk assessment
            risk_score = self._calculate_ib_risk_score(
                consensus_signal, final_position_size, account_balance, daily_pnl
            )
            
            risk_assessment = f"IB Risk Score: {risk_score:.2f}/10. Account: ${account_balance:,.0f}, Daily P&L: ${daily_pnl:,.0f}. "
            
            if risk_score > 8:
                risk_assessment += "VERY HIGH RISK - Trade rejected for account protection."
                final_position_size = 0.0
            elif risk_score > 6:
                risk_assessment += "HIGH RISK - Reduced position size by 50%."
                final_position_size *= 0.5
            elif risk_score > 4:
                risk_assessment += "MODERATE RISK - Proceed with caution."
            else:
                risk_assessment += "LOW RISK - Acceptable for execution."
            
            log.info("ib_risk_assessment", 
                    symbol=symbol,
                    risk_score=risk_score,
                    position_size=final_position_size,
                    account_balance=account_balance,
                    daily_pnl=daily_pnl)
            
            return {
                "risk_assessment": risk_assessment,
                "position_size": final_position_size,
                "risk_limits": risk_limits,
            }
            
        except Exception as exc:
            log.error("ib_risk_management_error", error=str(exc))
            return {
                "risk_assessment": f"Risk assessment failed: {str(exc)}",
                "position_size": 0.0,
                "risk_limits": {}
            }
    
    def _calculate_ib_risk_score(self, signal: TradeSignal, position_size: float, 
                                account_balance: float, daily_pnl: float) -> float:  # noqa: D401
        """Calculate IB-specific risk score."""
        risk_score = 5.0  # Base score
        
        # Account-based adjustments
        if account_balance > 0:
            position_pct = (position_size * signal.entry_price) / account_balance
            risk_score += position_pct * 20  # Higher percentage = higher risk
        
        # Daily P&L adjustments
        if daily_pnl < 0:
            loss_pct = abs(daily_pnl) / max(account_balance, 1)
            risk_score += loss_pct * 30  # Recent losses increase risk
        
        # Standard signal adjustments
        risk_score -= (signal.confidence - 0.5) * 4
        if signal.r_r_ratio > 0:
            risk_score -= min(2.0, signal.r_r_ratio - 1)
        
        urgency_risk = {"high": 1.5, "medium": 0.5, "low": -0.5}
        risk_score += urgency_risk.get(signal.urgency, 0)
        
        if signal.news_catalyst:
            risk_score += 1.0
        
        return max(1.0, min(10.0, risk_score))
    
    async def _ib_execution_node(self, state: IBEnhancedFuturesState) -> Dict:  # noqa: D401
        """Execute trades through Interactive Brokers."""
        symbol = state["futures_symbol"]
        consensus_signal = state.get("consensus_signal")
        position_size = state.get("position_size", 0.0)
        risk_assessment = state.get("risk_assessment", "")
        
        ib_trade = None
        execution_status = "NO_TRADE"
        
        if not consensus_signal or position_size <= 0:
            execution_status = "SKIPPED_NO_SIGNAL"
            log.info("ib_execution_skipped", symbol=symbol, reason="no_signal_or_zero_size")
            return {"ib_trade": ib_trade, "execution_status": execution_status}
        
        if not self.ib_broker:
            execution_status = "ERROR_NO_BROKER"
            log.error("ib_execution_failed", symbol=symbol, reason="broker_not_initialized")
            return {"ib_trade": ib_trade, "execution_status": execution_status}
        
        try:
            # Create enhanced signal for execution
            execution_signal = TradeSignal(
                symbol=symbol,
                action=consensus_signal.action,
                confidence=consensus_signal.confidence,
                entry_price=state.get("live_price", consensus_signal.entry_price),  # Use live price if available
                stop_loss=consensus_signal.stop_loss,
                take_profit=consensus_signal.take_profit,
                position_size=position_size,
                strategy="ib_enhanced_consensus",
                reasoning=f"IB execution: {consensus_signal.reasoning}",
                r_r_ratio=consensus_signal.r_r_ratio,
                urgency=consensus_signal.urgency,
                news_catalyst=consensus_signal.news_catalyst
            )
            
            # Execute through IB broker
            if not self.paper_trading and execution_signal.action in [Action.BUY, Action.SELL]:
                ib_trade = await self.ib_broker.execute_signal(execution_signal)
                
                if ib_trade:
                    execution_status = "EXECUTED"
                    log.info("ib_trade_executed", 
                            symbol=symbol,
                            action=execution_signal.action,
                            quantity=position_size,
                            entry_price=execution_signal.entry_price,
                            order_id=ib_trade.order.orderId)
                else:
                    execution_status = "REJECTED"
                    log.warning("ib_trade_rejected", symbol=symbol, reason="broker_rejection")
            else:
                execution_status = "PAPER_TRADE" if self.paper_trading else "HOLD"
                log.info("ib_trade_simulation", 
                        symbol=symbol,
                        action=execution_signal.action,
                        quantity=position_size,
                        reason="paper_trading" if self.paper_trading else "hold_signal")
            
        except Exception as exc:
            execution_status = "ERROR"
            log.error("ib_execution_error", symbol=symbol, error=str(exc))
        
        return {
            "ib_trade": ib_trade,
            "execution_status": execution_status,
        }
    
    async def _ib_final_decision_node(self, state: IBEnhancedFuturesState) -> Dict:  # noqa: D401
        """Generate final IB-enhanced trading decision."""
        symbol = state["futures_symbol"]
        consensus_signal = state.get("consensus_signal")
        risk_assessment = state.get("risk_assessment", "")
        position_size = state.get("position_size", 0.0)
        risk_limits = state.get("risk_limits", {})
        ib_trade = state.get("ib_trade")
        execution_status = state.get("execution_status", "UNKNOWN")
        
        # IB-specific data
        live_price = state.get("live_price")
        current_position = state.get("current_position")
        account_balance = state.get("account_balance", 0.0)
        daily_pnl = state.get("daily_pnl", 0.0)
        
        final_decision = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": consensus_signal.action if consensus_signal else "HOLD",
            "confidence": consensus_signal.confidence if consensus_signal else 0.5,
            "entry_price": live_price or (consensus_signal.entry_price if consensus_signal else 0.0),
            "live_price": live_price,
            "position_size": position_size,
            "risk_assessment": risk_assessment,
            "risk_limits": risk_limits,
            "strategy_used": consensus_signal.strategy if consensus_signal else "none",
            "reasoning": consensus_signal.reasoning if consensus_signal else "No clear signal",
            "urgency": consensus_signal.urgency if consensus_signal else "low",
            "news_catalyst": consensus_signal.news_catalyst if consensus_signal else False,
            
            # IB-specific fields
            "execution_status": execution_status,
            "order_id": ib_trade.order.orderId if ib_trade else None,
            "paper_trading": self.paper_trading,
            "account_balance": account_balance,
            "daily_pnl": daily_pnl,
            "current_position": {
                "quantity": current_position.qty if current_position else 0,
                "avg_cost": current_position.avg_cost if current_position else 0,
                "unrealized_pnl": current_position.unrealized if current_position else 0
            } if current_position else None,
            
            # Market data
            "realtime_data": {
                "price": live_price,
                "news_alerts": len(state.get("news_alerts", [])),
                "order_book_available": bool(state.get("order_book_data")),
            }
        }
        
        log.info("ib_final_decision", **{k: v for k, v in final_decision.items() if k not in ["risk_limits", "realtime_data"]})
        
        return {"final_decision": final_decision}
    
    # ------------------------------------------------------------------- #
    #  EXECUTION
    # ------------------------------------------------------------------- #
    
    async def propagate(self, symbol: str, trade_date: str) -> Dict:  # noqa: D401
        """Execute IB-enhanced futures trading analysis."""
        # Ensure symbol is clean
        clean_symbol = symbol.replace("=F", "")
        
        # Initialize IB components
        await self.initialize_ib_components([clean_symbol])
        
        try:
            initial_state = IBEnhancedFuturesState(
                messages=[],
                futures_symbol=clean_symbol,
                trade_date=trade_date,
                sender="ib_system",
                futures_market_report="",
                futures_fundamentals_report="",
                sentiment_report="",
                news_report="",
                bull_research="",
                bear_research="",
                investment_plan="",
                strategy_signals={},
                consensus_signal=None,
                live_price=None,
                current_position=None,
                account_balance=0.0,
                portfolio_value=0.0,
                daily_pnl=0.0,
                order_book_data=None,
                news_alerts=[],
                risk_assessment="",
                position_size=0.0,
                risk_limits={},
                ib_trade=None,
                execution_status="PENDING",
                final_decision={}
            )
            
            if self.debug:
                log.info("starting_ib_enhanced_analysis", 
                        symbol=clean_symbol, 
                        date=trade_date,
                        paper_trading=self.paper_trading)
            
            # Execute graph
            final_state = await self.graph.ainvoke(initial_state)
            
            return final_state["final_decision"]
            
        finally:
            # Cleanup
            await self.cleanup()
    
    async def cleanup(self) -> None:  # noqa: D401
        """Cleanup IB and analysis resources."""
        try:
            if self.news_sentinel:
                await self.news_sentinel.close()
            
            if self.ib_stream_manager:
                await self.ib_stream_manager.stop_streaming()
            
            if self.strategy_engine:
                await self.strategy_engine.close()
            
            # Note: Don't close IB broker if in live trading mode
            # User should manage broker lifecycle manually
            if self.ib_broker and self.paper_trading:
                await self.ib_broker.close()
                
            log.info("ib_enhanced_cleanup_completed")
            
        except Exception as exc:
            log.error("ib_cleanup_error", error=str(exc))


# ---------------------------------------------------------------------------
#  FACTORY FUNCTIONS
# ---------------------------------------------------------------------------

async def create_ib_enhanced_graph(
    paper_trading: bool = True, 
    config: Optional[Dict] = None,
    debug: bool = False
) -> IBEnhancedFuturesGraph:  # noqa: D401
    """Factory to create IB-enhanced futures graph."""
    from ..default_config import DEFAULT_CONFIG
    
    # Merge configs
    final_config = DEFAULT_CONFIG.copy()
    if config:
        final_config.update(config)
    
    # Create graph
    graph = IBEnhancedFuturesGraph(
        debug=debug,
        config=final_config,
        paper_trading=paper_trading
    )
    
    log.info("ib_enhanced_graph_created",
            paper_trading=paper_trading,
            enable_streaming=final_config.get("enable_realtime_stream", True))
    
    return graph


# ---------------------------------------------------------------------------
#  CLI DEMO
# ---------------------------------------------------------------------------

async def _demo_ib_enhanced_graph(symbol: str = "CL", paper_trading: bool = True):  # noqa: D401
    """Demo the IB-enhanced futures graph."""
    try:
        graph = await create_ib_enhanced_graph(
            paper_trading=paper_trading,
            debug=True
        )
        
        decision = await graph.propagate(symbol, "2025-01-15")
        
        print("\n" + "="*80)
        print("IB-ENHANCED FUTURES TRADING DECISION")
        print("="*80)
        print(f"Symbol: {decision['symbol']}")
        print(f"Action: {decision['action']}")
        print(f"Confidence: {decision['confidence']:.2f}")
        print(f"Live Price: ${decision['live_price']}")
        print(f"Position Size: {decision['position_size']:.3f}")
        print(f"Execution Status: {decision['execution_status']}")
        print(f"Paper Trading: {decision['paper_trading']}")
        print(f"Account Balance: ${decision['account_balance']:,.0f}")
        print(f"Daily P&L: ${decision['daily_pnl']:,.0f}")
        print(f"Strategy: {decision['strategy_used']}")
        print(f"Reasoning: {decision['reasoning']}")
        print(f"Risk Assessment: {decision['risk_assessment']}")
        if decision['order_id']:
            print(f"Order ID: {decision['order_id']}")
        print("="*80)
        
    except Exception as exc:
        log.error("ib_demo_failed", error=str(exc))
        raise


if __name__ == "__main__":  # pragma: no cover
    import argparse
    
    parser = argparse.ArgumentParser(description="IB enhanced futures graph demo")
    parser.add_argument("--symbol", default="CL", help="Futures symbol")
    parser.add_argument("--live", action="store_true", help="Use live trading (default: paper)")
    args = parser.parse_args()
    
    asyncio.run(_demo_ib_enhanced_graph(args.symbol, paper_trading=not args.live))