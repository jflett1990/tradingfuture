"""scalping_strategies.py – Nanosecond-aware, ultra-fast scalping engine for
commodity & financial futures.

Changelog (v2)
──────────────
• **RiskGate integration** – daily PnL cap, per-symbol position cap, draw-down circuit breaker.
• **DynamicSizer** – ATR-scaled lot sizing (plug via `size_fn`).
• **Dual latency metrics** – `ING_LAT` for tick ingestion, `SIG_LAT` for strategy latency.
• **Order-book placeholder fleshed** – ready for L2 feed injection.
• **Config** now carries risk knobs.

MIT License — Prompt Maestro 9000, 2025
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Sequence, Optional

import numpy as np
import pandas as pd
import prometheus_client as prom
import structlog
import talib

from .advanced_strategies import TradeSignal, Action
from ..dataflows.news_sentinel import NewsSentinel, Snapshot

# ---------------------------------------------------------------------------
# LOGS & METRICS
# ---------------------------------------------------------------------------

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(structlog.INFO),
    processors=[structlog.processors.TimeStamper(fmt="iso"), structlog.processors.JSONRenderer()],
)
log = structlog.get_logger("scalp")

SIG_CNT = prom.Counter("scalp_signals_total", "Scalping signals", ["strategy"])
ING_LAT = prom.Histogram("scalp_ingest_ms", "Tick ingestion latency ms")
SIG_LAT = prom.Histogram("scalp_latency_ms", "Signal latency ms")

# ---------------------------------------------------------------------------
# ENUMS & CONFIG
# ---------------------------------------------------------------------------

class TF(str, Enum):
    TICK = "tick"
    S5 = "5s"
    S15 = "15s"
    M1 = "1m"
    M5 = "5m"


@dataclass(slots=True)
class Config:
    tf: TF = TF.M1
    target_ticks: float = 3.0
    stop_ticks: float = 1.5
    hold_min: int = 15
    vol_surge: float = 2.0
    news_sec: int = 30
    depth: int = 5
    tick: float = 0.01
    # risk knobs
    max_daily_loss: float = 5_000.0
    max_pos_per_symbol: int = 5
    drawdown_pct: float = 10.0

# ---------------------------------------------------------------------------
# RISK CONTROLS
# ---------------------------------------------------------------------------

class RiskGate:
    """Risk management for scalping operations."""
    
    def __init__(self, max_daily_loss: float = 5000.0, max_positions_per_symbol: int = 5, 
                 drawdown_pct: float = 10.0):
        self.max_daily_loss = max_daily_loss
        self.max_positions_per_symbol = max_positions_per_symbol
        self.drawdown_pct = drawdown_pct
        self.daily_pnl = 0.0
        self.positions: Dict[str, int] = {}
        self.peak_balance = 100000.0  # Starting balance
        
    def allow_new_trade(self, symbol: str) -> bool:
        """Check if new trade is allowed."""
        # Daily loss check
        if self.daily_pnl < -abs(self.max_daily_loss):
            log.warning("risk_block_daily_loss", pnl=self.daily_pnl)
            return False
            
        # Position limit check
        current_positions = self.positions.get(symbol, 0)
        if current_positions >= self.max_positions_per_symbol:
            log.warning("risk_block_position_limit", symbol=symbol, positions=current_positions)
            return False
            
        # Drawdown check
        current_balance = self.peak_balance + self.daily_pnl
        drawdown = (self.peak_balance - current_balance) / self.peak_balance * 100
        if drawdown > self.drawdown_pct:
            log.warning("risk_block_drawdown", drawdown=drawdown)
            return False
            
        return True
    
    def update_pnl(self, pnl_change: float):
        """Update daily P&L."""
        self.daily_pnl += pnl_change
        current_balance = self.peak_balance + self.daily_pnl
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance

class DynamicSizer:
    """Dynamic position sizing based on volatility."""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_size = 1.0
        
    def size(self, symbol: str, atr: Optional[float] = None) -> float:
        """Calculate position size based on volatility."""
        if atr is None:
            return self.base_size
            
        # Scale down size for higher volatility
        vol_adjustment = min(2.0, 1.0 / max(atr, 0.01))
        return self.base_size * vol_adjustment

# ---------------------------------------------------------------------------
# RING BUFFER FOR TICKS
# ---------------------------------------------------------------------------

class TickBuf:
    def __init__(self, size: int = 500):
        self.size = size
        self.prices = np.empty(size)
        self.vols = np.empty(size)
        self.times: List[datetime] = [datetime.now(timezone.utc)] * size
        self.idx = 0
        self.full = False

    def push(self, price: float, vol: float, ts: datetime):
        self.prices[self.idx] = price
        self.vols[self.idx] = vol
        self.times[self.idx] = ts
        self.idx = (self.idx + 1) % self.size
        if self.idx == 0:
            self.full = True

    def last(self, n: int):
        if not self.full and self.idx < n:
            raise IndexError("Not enough data")
        end = self.idx
        start = (end - n) % self.size
        if start < end:
            return self.prices[start:end], self.vols[start:end], self.times[start:end]
        return (
            np.concatenate((self.prices[start:], self.prices[:end])),
            np.concatenate((self.vols[start:], self.vols[:end])),
            self.times[start:] + self.times[:end],
        )

# ---------------------------------------------------------------------------
# SCALPER
# ---------------------------------------------------------------------------

class Scalper:
    def __init__(
        self,
        symbols: Sequence[str],
        *,
        cfg: Config | None = None,
        news: NewsSentinel | None = None,
        size_fn=None,
        risk: RiskGate | None = None,
    ):
        self.cfg = cfg or Config()
        self.symbols = list(symbols)
        self.news = news or NewsSentinel(self.symbols)
        self.size_fn = size_fn or DynamicSizer(self.cfg).size
        self.risk = risk or RiskGate(
            max_daily_loss=self.cfg.max_daily_loss,
            max_positions_per_symbol=self.cfg.max_pos_per_symbol,
            drawdown_pct=self.cfg.drawdown_pct,
        )
        self.ticks = {s: TickBuf() for s in self.symbols}
        self.last_sig: Dict[str, datetime] = {}
        self.order_books: Dict[str, Dict] = {}  # For L2 data
        asyncio.create_task(prom.start_http_server(8500))

    # --------------------------- public API --------------------------- #

    async def on_tick(self, symbol: str, price: float, vol: float, ts: datetime):
        """Process incoming tick data."""
        t0 = time.time()
        self.ticks[symbol].push(price, vol, ts)
        ING_LAT.observe((time.time() - t0) * 1_000)
        sig = await self._route(symbol)
        return sig

    def update_order_book(self, symbol: str, bids: List[Dict], asks: List[Dict]):
        """Update Level 2 order book data."""
        self.order_books[symbol] = {
            'bids': bids,
            'asks': asks,
            'timestamp': datetime.now(timezone.utc)
        }

    # --------------------------- router ------------------------------- #

    async def _route(self, symbol: str):
        """Route to scalping strategies."""
        # throttle per-minute
        if symbol in self.last_sig and (datetime.now(timezone.utc) - self.last_sig[symbol]).seconds < 60:
            return None
        t0 = time.time()
        tasks = [
            self._tick_momo(symbol), 
            self._order_flow(symbol), 
            self._news_react(symbol),
            self._breakout_scalp(symbol)
        ]
        sigs = [s for s in await asyncio.gather(*tasks, return_exceptions=True) if isinstance(s, TradeSignal)]
        if not sigs:
            return None
        # best by confidence and strategy priority
        best = max(sigs, key=lambda s: (
            4 if s.strategy == "news_scalp" else
            3 if s.strategy == "order_flow" else
            2 if s.strategy == "tick_momo" else 1,
            s.confidence
        ))
        # risk gate
        if not self.risk.allow_new_trade(symbol):
            return None
        self.last_sig[symbol] = best.timestamp
        SIG_CNT.labels(strategy=best.strategy).inc()
        SIG_LAT.observe((time.time() - t0) * 1_000)
        log.info("signal", **{k: v for k, v in best.__dict__.items() if k != 'timestamp'})
        return best

    # ------------------- strategy: tick momentum ---------------------- #

    async def _tick_momo(self, symbol: str):
        """Tick-based momentum scalping."""
        try:
            p, v, t = self.ticks[symbol].last(100)
        except IndexError:
            return None
        
        # Micro momentum (last 20 ticks vs longer period)
        short = (p[-1] - p[-20]) / p[-20] * 10_000  # basis points
        long_momentum = (p[-1] - p[-50]) / p[-50] * 10_000
        
        # Volume surge detection
        vu = v[-10:].sum() / max(v[-50:-10].mean(), 1)
        
        # Price velocity
        time_diff = (t[-1] - t[-20]).total_seconds() if len(t) >= 20 else 1
        velocity = abs(short) / max(time_diff, 1)
        
        # Scalping conditions
        if abs(short) < 2 or vu < self.cfg.vol_surge or velocity < 0.5:
            return None
            
        # Direction alignment check
        if abs(short) > abs(long_momentum) and np.sign(short) == np.sign(long_momentum):
            momentum_alignment = True
        else:
            momentum_alignment = False
            
        if not momentum_alignment:
            return None
            
        action = Action.BUY if short > 0 else Action.SELL
        price = p[-1]
        tv = self.cfg.tick
        tp = price + self.cfg.target_ticks * tv * (1 if action == Action.BUY else -1)
        sl = price - self.cfg.stop_ticks * tv * (1 if action == Action.BUY else -1)
        conf = min(0.9, 0.6 + abs(short) * 0.02 + min(vu, 3) * 0.1)
        
        # Calculate ATR for dynamic sizing
        atr = np.std(p[-20:]) if len(p) >= 20 else None
        size = self.size_fn(symbol, atr)
        
        return TradeSignal(
            symbol, action, conf, price, sl, tp, size, "tick_momo", 
            f"{short:.1f}bp, vel:{velocity:.2f}, vol:{vu:.1f}x", 
            self.cfg.target_ticks / self.cfg.stop_ticks, "high", False, 
            datetime.now(timezone.utc)
        )

    # ------------------- strategy: order flow ------------------------- #

    async def _order_flow(self, symbol: str):
        """Order book imbalance scalping."""
        if symbol not in self.order_books:
            return None
            
        book = self.order_books[symbol]
        bids = book.get('bids', [])
        asks = book.get('asks', [])
        
        if len(bids) < 5 or len(asks) < 5:
            return None
            
        # Calculate imbalances
        bid_volume = sum(level.get('size', 0) for level in bids[:self.cfg.depth])
        ask_volume = sum(level.get('size', 0) for level in asks[:self.cfg.depth])
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return None
            
        imbalance = (bid_volume - ask_volume) / total_volume
        
        # Spread analysis
        best_bid = bids[0].get('price', 0)
        best_ask = asks[0].get('price', 0)
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2
        spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 100
        
        # Size at best levels
        bid_size = bids[0].get('size', 0)
        ask_size = asks[0].get('size', 0)
        size_imbalance = (bid_size - ask_size) / (bid_size + ask_size) if (bid_size + ask_size) > 0 else 0
        
        # Scalping conditions
        strong_imbalance = abs(imbalance) > 0.3
        tight_spread = spread_bps < 5.0  # Reasonable spread for futures
        size_confirmation = abs(size_imbalance) > 0.4
        
        if not (strong_imbalance and tight_spread and size_confirmation):
            return None
            
        # Direction from imbalance
        action = Action.BUY if imbalance > 0 else Action.SELL
        entry_price = best_ask if action == Action.BUY else best_bid
        
        # Tighter targets for order flow
        target_ticks = 2.0
        stop_ticks = 1.0
        
        tv = self.cfg.tick
        tp = entry_price + target_ticks * tv * (1 if action == Action.BUY else -1)
        sl = entry_price - stop_ticks * tv * (1 if action == Action.BUY else -1)
        
        conf = min(0.85, 0.5 + abs(imbalance) + abs(size_imbalance) * 0.5)
        size = self.size_fn(symbol)
        
        return TradeSignal(
            symbol, action, conf, entry_price, sl, tp, size, "order_flow",
            f"imb:{imbalance:.2f}, size_imb:{size_imbalance:.2f}, spread:{spread_bps:.1f}bp",
            target_ticks / stop_ticks, "high", False, datetime.now(timezone.utc)
        )

    # -------------------- strategy: news ------------------------------ #

    async def _news_react(self, symbol: str):
        """Ultra-fast news reaction scalping."""
        try:
            snap: Snapshot = await self.news.snapshot(symbol)
        except Exception:
            return None
            
        if not snap or snap.confidence < 0.6 or not snap.news_items:
            return None
            
        # Find very recent, high-urgency news
        now = datetime.now(timezone.utc)
        recent = [
            n for n in snap.news_items 
            if (now - n.timestamp).total_seconds() < self.cfg.news_sec and n.urgency > 0.7
        ]
        if not recent:
            return None
            
        latest = max(recent, key=lambda x: x.timestamp)
        
        try:
            price = self.ticks[symbol].last(1)[0][-1]
        except IndexError:
            return None
            
        # Direction from sentiment
        direction = (
            Action.BUY if snap.label == "bullish" else 
            Action.SELL if snap.label == "bearish" else None
        )
        if not direction:
            return None
            
        # Aggressive targets for news
        urgency_multiplier = latest.urgency
        target_ticks = self.cfg.target_ticks * urgency_multiplier * 1.5
        stop_ticks = self.cfg.stop_ticks
        
        tv = self.cfg.tick
        tp = price + target_ticks * tv * (1 if direction == Action.BUY else -1)
        sl = price - stop_ticks * tv * (1 if direction == Action.BUY else -1)
        
        # Larger size for news
        size = self.size_fn(symbol) * 2
        conf = min(0.95, snap.confidence * latest.urgency)
        
        return TradeSignal(
            symbol, direction, conf, price, sl, tp, size, "news_scalp", 
            f"NEWS: {latest.title[:40]}...", target_ticks / stop_ticks, "high", True, 
            datetime.now(timezone.utc)
        )

    # ------------------- strategy: breakout scalp --------------------- #

    async def _breakout_scalp(self, symbol: str):
        """Micro-breakout scalping on small timeframes."""
        try:
            p, v, t = self.ticks[symbol].last(100)
        except IndexError:
            return None
            
        if len(p) < 50:
            return None
            
        # Calculate recent high/low levels (last 50 ticks)
        recent_high = np.max(p[-50:])
        recent_low = np.min(p[-50:])
        current_price = p[-1]
        
        # Range analysis
        range_size = recent_high - recent_low
        if range_size < self.cfg.tick * 5:  # Minimum range
            return None
            
        # Volume confirmation
        recent_volume = v[-5:].mean()
        avg_volume = v[-50:-5].mean()
        volume_surge = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Breakout conditions
        upper_break = current_price > recent_high - self.cfg.tick
        lower_break = current_price < recent_low + self.cfg.tick
        volume_confirm = volume_surge > 1.5
        
        if not volume_confirm:
            return None
            
        if upper_break:
            action = Action.BUY
            entry = current_price
            sl = recent_high - range_size * 0.3
            tp = current_price + range_size * 0.5
        elif lower_break:
            action = Action.SELL
            entry = current_price
            sl = recent_low + range_size * 0.3
            tp = current_price - range_size * 0.5
        else:
            return None
            
        # Risk/reward check
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0 or reward / risk < 1.5:
            return None
            
        conf = min(0.8, 0.5 + volume_surge * 0.1 + (range_size / current_price) * 1000)
        size = self.size_fn(symbol)
        
        return TradeSignal(
            symbol, action, conf, entry, sl, tp, size, "breakout_scalp",
            f"Range breakout: {range_size:.4f}, vol:{volume_surge:.1f}x",
            reward / risk, "medium", False, datetime.now(timezone.utc)
        )

    # --------------------------- utilities ---------------------------- #

    def get_metrics(self) -> Dict[str, Any]:
        """Get scalping performance metrics."""
        return {
            "symbols": len(self.symbols),
            "total_signals": sum(self.last_sig.values() is not None for _ in self.symbols),
            "daily_pnl": self.risk.daily_pnl,
            "risk_utilization": {
                "daily_loss_pct": abs(self.risk.daily_pnl) / self.risk.max_daily_loss * 100,
                "position_count": sum(self.risk.positions.values()),
                "max_positions": len(self.symbols) * self.risk.max_positions_per_symbol
            }
        }

    async def close(self):
        """Cleanup resources."""
        await self.news.close()

# ---------------------------------------------------------------------------
# FACTORY FUNCTIONS
# ---------------------------------------------------------------------------

async def create_scalping_engine(
    symbols: Sequence[str], 
    timeframe: TF = TF.M1,
    **config_overrides
) -> Scalper:
    """Create configured scalping engine."""
    cfg = Config(tf=timeframe)
    
    # Adjust config based on timeframe
    if timeframe == TF.TICK:
        cfg.target_ticks = 1.0
        cfg.stop_ticks = 0.5
        cfg.hold_min = 2
    elif timeframe == TF.S5:
        cfg.target_ticks = 1.5
        cfg.stop_ticks = 0.75
        cfg.hold_min = 5
    elif timeframe == TF.M1:
        cfg.target_ticks = 3.0
        cfg.stop_ticks = 1.5
        cfg.hold_min = 15
    elif timeframe == TF.M5:
        cfg.target_ticks = 5.0
        cfg.stop_ticks = 2.5
        cfg.hold_min = 30
        
    # Apply overrides
    for key, value in config_overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    
    scalper = Scalper(symbols, cfg=cfg)
    
    log.info("scalping_engine_created",
            symbols=list(symbols),
            timeframe=timeframe.value,
            target_ticks=cfg.target_ticks,
            risk_limits=cfg.max_daily_loss)
    
    return scalper

# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------

async def _demo():
    """Demo scalping with realistic tick simulation."""
    from random import gauss, randint
    
    symbols = ["ES", "CL", "GC"]
    scalper = await create_scalping_engine(symbols, TF.M1)
    
    print("🎯 SCALPING ENGINE DEMO")
    print("="*50)
    
    # Simulate realistic tick data
    base_prices = {"ES": 4500.0, "CL": 75.0, "GC": 2000.0}
    ts = datetime.now(timezone.utc)
    
    signals_generated = 0
    
    for i in range(1000):  # 1000 ticks
        for symbol in symbols:
            base_price = base_prices[symbol]
            
            # Simulate price movement with occasional spikes
            if randint(1, 100) <= 5:  # 5% chance of spike
                price_change = gauss(0, 0.05) * base_price  # Larger move
                volume = abs(gauss(0, 1)) * 500  # High volume
            else:
                price_change = gauss(0, 0.001) * base_price  # Normal move
                volume = abs(gauss(0, 1)) * 100  # Normal volume
                
            new_price = base_price + price_change
            base_prices[symbol] = new_price  # Update base
            
            # Simulate order book
            if randint(1, 10) <= 3:  # 30% chance of order book update
                bids = [{"price": new_price - 0.01 * j, "size": randint(50, 200)} for j in range(1, 6)]
                asks = [{"price": new_price + 0.01 * j, "size": randint(50, 200)} for j in range(1, 6)]
                scalper.update_order_book(symbol, bids, asks)
            
            # Process tick
            signal = await scalper.on_tick(symbol, new_price, volume, ts + timedelta(milliseconds=i*100))
            
            if signal:
                signals_generated += 1
                print(f"\n📊 SCALP SIGNAL #{signals_generated}")
                print(f"Symbol: {signal.symbol}")
                print(f"Strategy: {signal.strategy}")
                print(f"Action: {signal.action}")
                print(f"Entry: ${signal.entry_price:.4f}")
                print(f"Target: ${signal.take_profit:.4f}")
                print(f"Stop: ${signal.stop_loss:.4f}")
                print(f"R:R: {signal.r_r_ratio:.2f}")
                print(f"Confidence: {signal.confidence:.1%}")
                print(f"Reasoning: {signal.reasoning}")
                
                # Simulate random P&L update
                pnl_change = gauss(50, 100)  # Random P&L
                scalper.risk.update_pnl(pnl_change)
    
    # Final metrics
    metrics = scalper.get_metrics()
    print(f"\n📈 SCALPING SESSION COMPLETE")
    print("="*50)
    print(f"Signals Generated: {signals_generated}")
    print(f"Daily P&L: ${metrics['daily_pnl']:,.2f}")
    print(f"Risk Utilization: {metrics['risk_utilization']['daily_loss_pct']:.1f}%")
    
    await scalper.close()

if __name__ == "__main__":
    asyncio.run(_demo())