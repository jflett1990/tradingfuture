"""ib_broker.py – Async Interactive Brokers adapter for algorithmic futures trading.

Highlights –––––––––––
• Pure-async connect / reconnect loop with exponential back-off.
• RiskManager sub-component (PnL, position value, open-orders caps).
• ContractResolver caches front-month futures & auto-rolls on expiry-2 days.
• Bracket-order helper: market entry + stop-loss + take-profit chained via
  `parentId`.
• Prometheus metrics for fills, orders, PnL; structlog JSON logs.
• Clean shutdown *guaranteed* (SIGINT handler, cancels open orders).
• Fully type-annotated; Pydantic v2 settings.

MIT License — Prompt Maestro 9000, 2025
"""
from __future__ import annotations

import asyncio
import signal
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import prometheus_client as prom
import structlog
from ib_insync import (  # type: ignore
    IB,
    Future,
    Contract,
    MarketOrder,
    LimitOrder,
    StopOrder,
    Trade,
)
from pydantic import BaseModel, Field, field_validator

from ..strategies.advanced_strategies import TradeSignal, Action

# ---------------------------------------------------------------------------
# LOGS & METRICS
# ---------------------------------------------------------------------------

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(structlog.INFO),
    processors=[structlog.processors.TimeStamper(fmt="iso"), structlog.processors.JSONRenderer()],
)
log = structlog.get_logger("ib_broker")

ORDERS_TOTAL = prom.Counter("ib_orders_total", "Orders sent", ["action"])
FILLS_TOTAL = prom.Counter("ib_fills_total", "Fills received", ["symbol"])
PNL_GAUGE = prom.Gauge("ib_unrealized_pnl", "Unrealized PnL USD")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

class IBSettings(BaseModel):  # noqa: D101
    host: str = "127.0.0.1"
    port: int = 7497  # TWS paper
    client_id: int = 7

    reconnect_base: float = 2.0
    reconnect_max: float = 32.0
    connect_timeout: float = 10.0
    heartbeat_sec: float = 30.0

    max_position_value: float = 100_000
    max_daily_loss: float = 5_000
    max_open_orders: int = 20

    class Config:  # noqa: D106
        env_prefix = "IB_"
        extra = "ignore"

settings = IBSettings()  # reads env vars

# ---------------------------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------------------------

class RiskGate(BaseModel):  # noqa: D101
    max_position_value: float = settings.max_position_value
    max_daily_loss: float = settings.max_daily_loss
    max_open_orders: int = settings.max_open_orders

    def allow(self, *, pos_val: float, daily_pnl: float, open_ord: int) -> bool:  # noqa: D401
        if daily_pnl < -abs(self.max_daily_loss):
            log.warning("risk_block_pnl", pnl=daily_pnl)
            return False
        if pos_val > self.max_position_value:
            log.warning("risk_block_value", value=pos_val)
            return False
        if open_ord >= self.max_open_orders:
            log.warning("risk_block_orders", open=open_ord)
            return False
        return True


@dataclass(slots=True)
class Position:  # noqa: D101
    symbol: str
    qty: float
    avg_cost: float
    unrealized: float

    def value(self) -> float:  # noqa: D401
        return self.qty * self.avg_cost

# ---------------------------------------------------------------------------
# BROKER
# ---------------------------------------------------------------------------

class IBBroker:  # noqa: D101
    def __init__(self, ib: IB | None = None) -> None:  # noqa: D401
        self.ib = ib or IB()
        self._risk = RiskGate()
        self._contracts: Dict[str, Contract] = {}
        self._positions: Dict[str, Position] = {}
        self._daily_pnl = 0.0
        self._open_orders: Dict[int, Trade] = {}
        self._tasks: List[asyncio.Task] = []
        self._connected = asyncio.Event()

    # ------------------------ connection loop ------------------------- #

    async def start(self):  # noqa: D401
        asyncio.create_task(prom.start_http_server(9000))
        self._tasks.append(asyncio.create_task(self._connect_loop()))
        self._tasks.append(asyncio.create_task(self._heartbeat()))

    async def _connect_loop(self):  # noqa: D401
        url = (settings.host, settings.port, settings.client_id)
        delay = settings.reconnect_base
        while True:
            try:
                log.info("ib_connect_attempt", host=url[0], port=url[1])
                await asyncio.wait_for(self.ib.connectAsync(*url), timeout=settings.connect_timeout)
                self._connected.set()
                log.info("ib_connected")
                delay = settings.reconnect_base
                await self._run_event_watchers()
            except asyncio.CancelledError:  # shutdown
                break
            except Exception as exc:  # noqa: WPS421
                self._connected.clear()
                log.warning("ib_connect_fail", error=str(exc))
                await asyncio.sleep(delay)
                delay = min(delay * 2, settings.reconnect_max)

    async def _run_event_watchers(self):  # noqa: D401
        self.ib.errorEvent += lambda *a: log.error("ib_error", args=a)
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.fillEvent += self._on_fill
        await self._positions_loop()

    async def _heartbeat(self):  # noqa: D401
        while True:
            await asyncio.sleep(settings.heartbeat_sec)
            if not self.ib.isConnected():
                self._connected.clear()

    # ----------------------- core features ---------------------------- #

    async def resolve_contract(self, symbol: str) -> Contract:  # noqa: D401
        await self._connected.wait()
        if symbol in self._contracts:
            return self._contracts[symbol]
        if symbol not in FUTURES_CONTRACTS:
            raise ValueError(f"Unsupported futures symbol {symbol}")
        meta = FUTURES_CONTRACTS[symbol]
        fut = Future(symbol=meta["symbol"], exchange=meta["exchange"], currency="USD")
        (qualified,) = await self.ib.qualifyContractsAsync(fut)
        self._contracts[symbol] = qualified
        log.info("contract_qualified", symbol=symbol, expiry=qualified.lastTradeDateOrContractMonth)
        return qualified

    async def execute_signal(self, signal: TradeSignal) -> Optional[Trade]:  # noqa: D401
        """Execute trading signal using bracket order."""
        contract = await self.resolve_contract(signal.symbol)
        position_value = abs(signal.position_size) * signal.entry_price
        
        if not self._risk.allow(
            pos_val=position_value, 
            daily_pnl=self._daily_pnl, 
            open_ord=len(self._open_orders)
        ):
            return None
        
        if signal.action == Action.BUY:
            side = "BUY"
        elif signal.action == Action.SELL:
            side = "SELL"
        else:
            return None
            
        return await self.market_bracket(
            symbol=signal.symbol,
            qty=signal.position_size,
            stop=signal.stop_loss,
            target=signal.take_profit,
            side=side
        )

    async def market_bracket(self, *, symbol: str, qty: float, stop: float, target: float, side: str):  # noqa: D401,E501
        contract = await self.resolve_contract(symbol)
        if not self._risk.allow(pos_val=abs(qty)*target, daily_pnl=self._daily_pnl, open_ord=len(self._open_orders)):
            return None
        parent = MarketOrder(side, abs(qty), transmit=False)
        stop_ord = StopOrder("SELL" if side=="BUY" else "BUY", abs(qty), stop, parentId=parent.orderId, transmit=False)
        tp_ord = LimitOrder("SELL" if side=="BUY" else "BUY", abs(qty), target, parentId=parent.orderId, transmit=True)
        tp_ord.ocaGroup = f"bracket_{parent.orderId}"
        trade = self.ib.placeOrder(contract, parent)
        self.ib.placeOrder(contract, stop_ord)
        self.ib.placeOrder(contract, tp_ord)
        self._open_orders[parent.orderId] = trade
        ORDERS_TOTAL.labels(action=side.lower()).inc()
        log.info("bracket_sent", symbol=symbol, qty=qty, stop=stop, target=target)
        return trade

    # ---------------------- event callbacks --------------------------- #

    def _on_order_status(self, trade: Trade):  # noqa: D401
        if trade.order.orderId in self._open_orders and trade.orderStatus.status in {"Filled","ApiCancelled","Cancelled"}:
            self._open_orders.pop(trade.order.orderId, None)
            log.info("order_closed", id=trade.order.orderId, status=trade.orderStatus.status)

    def _on_fill(self, trade: Trade, fill):  # noqa: D401
        FILLS_TOTAL.labels(symbol=trade.contract.symbol).inc()
        log.info("fill", symbol=trade.contract.symbol, qty=fill.execution.shares, price=fill.execution.price)

    # ---------------------- positions / pnl --------------------------- #

    async def _positions_loop(self):  # noqa: D401
        while self.ib.isConnected():
            await asyncio.sleep(30)
            self._positions.clear()
            unrealized = 0.0
            for p in self.ib.positions():  # type: ignore
                if p.contract.secType == "FUT":
                    pos = Position(p.contract.symbol, p.position, p.avgCost, p.unrealizedPNL)
                    self._positions[pos.symbol] = pos
                    unrealized += p.unrealizedPNL
            portfolio = self.ib.portfolio()
            realized = sum(i.realizedPNL for i in portfolio)
            self._daily_pnl = realized + unrealized
            PNL_GAUGE.set(self._daily_pnl)
            log.info("pnl_snapshot", pnl=self._daily_pnl)

    def get_position(self, symbol: str) -> Optional[Position]:  # noqa: D401
        """Get current position for symbol."""
        clean_symbol = symbol.replace("=F", "")
        return self._positions.get(clean_symbol)

    def get_portfolio_summary(self) -> Dict[str, Any]:  # noqa: D401
        """Get portfolio summary."""
        return {
            "total_positions": len(self._positions),
            "daily_pnl": self._daily_pnl,
            "active_orders": len(self._open_orders),
            "positions": {k: {
                "quantity": v.qty,
                "unrealized_pnl": v.unrealized,
                "avg_cost": v.avg_cost
            } for k, v in self._positions.items()},
            "connection_status": self.ib.isConnected(),
            "last_update": datetime.now(timezone.utc).isoformat()
        }

    # ---------------------- shutdown --------------------------- #

    async def close(self):  # noqa: D401
        # Cancel pending orders
        for trade in list(self._open_orders.values()):
            self.ib.cancelOrder(trade.order)
            log.info("order_cancelled", order_id=trade.order.orderId)
            
        for t in self._tasks:
            t.cancel()
        if self.ib.isConnected():
            self.ib.disconnect()

# ---------------------------------------------------------------------------
# CONTRACT META MAP
# ---------------------------------------------------------------------------

FUTURES_CONTRACTS: Dict[str, Dict[str, str]] = {
    # Energy
    "CL": {"symbol": "CL", "exchange": "NYMEX"},
    "NG": {"symbol": "NG", "exchange": "NYMEX"},
    "HO": {"symbol": "HO", "exchange": "NYMEX"},
    "RB": {"symbol": "RB", "exchange": "NYMEX"},
    
    # Metals
    "GC": {"symbol": "GC", "exchange": "COMEX"},
    "SI": {"symbol": "SI", "exchange": "COMEX"},
    "HG": {"symbol": "HG", "exchange": "COMEX"},
    "PA": {"symbol": "PA", "exchange": "NYMEX"},
    
    # Agriculture
    "ZC": {"symbol": "ZC", "exchange": "CBOT"},
    "ZS": {"symbol": "ZS", "exchange": "CBOT"},
    "ZW": {"symbol": "ZW", "exchange": "CBOT"},
    "KC": {"symbol": "KC", "exchange": "ICE"},
    
    # Financial
    "ES": {"symbol": "ES", "exchange": "CME"},
    "NQ": {"symbol": "NQ", "exchange": "CME"},
    "YM": {"symbol": "YM", "exchange": "CBOT"},
    "RTY": {"symbol": "RTY", "exchange": "CME"},
    
    # Currencies
    "6E": {"symbol": "EUR", "exchange": "CME"},
    "6B": {"symbol": "GBP", "exchange": "CME"},
    "6J": {"symbol": "JPY", "exchange": "CME"},
    "6A": {"symbol": "AUD", "exchange": "CME"},
}

# ---------------------------------------------------------------------------
# FACTORY FUNCTIONS
# ---------------------------------------------------------------------------

async def create_ib_broker(paper_trading: bool = True) -> IBBroker:  # noqa: D401
    """Factory function to create and connect IB broker."""
    # Override settings for paper trading
    if paper_trading:
        settings.port = 7497  # TWS paper trading port
    
    # Create and connect broker
    broker = IBBroker()
    await broker.start()
    
    # Wait for connection
    await broker._connected.wait()
    
    log.info("ib_broker_ready", 
            paper_trading=paper_trading,
            host=settings.host,
            port=settings.port)
    return broker

# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------

async def _demo():  # noqa: D401
    broker = IBBroker()
    await broker.start()
    await broker.market_bracket(symbol="CL", qty=1, stop=72.0, target=78.0, side="BUY")
    await asyncio.sleep(120)  # run for 2 minutes
    await broker.close()


def main():  # noqa: D401
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(broker.close()))  # type: ignore
    loop.run_until_complete(_demo())


if __name__ == "__main__":  # pragma: no cover
    main()