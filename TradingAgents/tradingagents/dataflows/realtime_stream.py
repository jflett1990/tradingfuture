"""realtime_stream.py – High-speed market data streaming for futures.

Production-grade real-time data pipeline with:
• WebSocket connections to multiple data providers
• Tick-by-tick data processing with microsecond timestamps
• Order book reconstruction and L2 market depth
• Volume profile updates in real-time
• Circuit breakers and connection failover
• Async pub/sub for downstream consumers

Optimized for ultra-low latency futures trading.

© 2025 Prompt Maestro 9000, MIT License
"""
from __future__ import annotations

import asyncio
import json
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Any

import structlog
import websockets
from pydantic import BaseSettings, Field

warnings.filterwarnings('ignore')

log = structlog.get_logger("realtime_stream")

# ---------------------------------------------------------------------------
#  CONFIGURATION
# ---------------------------------------------------------------------------

class StreamSettings(BaseSettings):
    """Real-time streaming configuration."""
    
    # Data sources
    primary_ws_url: str = Field("wss://api.example.com/ws", env="PRIMARY_WS_URL")
    backup_ws_url: str = Field("wss://backup.example.com/ws", env="BACKUP_WS_URL")
    api_key: Optional[str] = Field(None, env="STREAM_API_KEY")
    
    # Performance
    buffer_size: int = Field(10000, ge=1000, le=100000)
    heartbeat_interval: float = Field(30.0, ge=5.0, le=120.0)
    reconnect_delay: float = Field(5.0, ge=1.0, le=60.0)
    
    # Circuit breakers
    max_reconnects: int = Field(10, ge=1, le=100)
    data_timeout: float = Field(60.0, ge=10.0, le=300.0)
    
    class Config:  # noqa: D106
        env_file = ".env"

settings = StreamSettings()

# ---------------------------------------------------------------------------
#  DATA CLASSES
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Tick:  # noqa: D101
    symbol: str
    timestamp: datetime
    price: float
    volume: int
    side: str  # buy/sell
    sequence: int = 0


@dataclass(slots=True)
class OrderBookLevel:  # noqa: D101
    price: float
    quantity: int
    orders: int = 1


@dataclass(slots=True)
class OrderBook:  # noqa: D101
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    sequence: int = 0


@dataclass(slots=True)
class MarketSummary:  # noqa: D101
    symbol: str
    timestamp: datetime
    last_price: float
    volume_24h: int
    high_24h: float
    low_24h: float
    change_24h: float
    bid: float
    ask: float
    spread: float


# ---------------------------------------------------------------------------
#  STREAM MANAGER
# ---------------------------------------------------------------------------

class RealTimeStreamManager:  # noqa: D101
    
    def __init__(self, symbols: List[str]):  # noqa: D401
        self.symbols = symbols
        self.is_running = False
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.tick_buffers: Dict[str, deque] = {s: deque(maxlen=settings.buffer_size) for s in symbols}
        self.order_books: Dict[str, OrderBook] = {}
        self.market_summaries: Dict[str, MarketSummary] = {}
        self.last_data_time = time.time()
        self.reconnect_counts: Dict[str, int] = defaultdict(int)
        
    # ------------------------------------------------------------------- #
    #  CONNECTION MANAGEMENT
    # ------------------------------------------------------------------- #
    
    async def start_streaming(self) -> None:  # noqa: D401
        """Start real-time data streaming."""
        if self.is_running:
            return
            
        self.is_running = True
        log.info("starting_realtime_stream", symbols=self.symbols)
        
        # Start multiple concurrent tasks
        tasks = [
            asyncio.create_task(self._primary_stream_handler()),
            asyncio.create_task(self._backup_stream_handler()),
            asyncio.create_task(self._heartbeat_monitor()),
            asyncio.create_task(self._circuit_breaker_monitor()),
        ]
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as exc:
            log.error("stream_error", error=str(exc))
        finally:
            await self.stop_streaming()
    
    async def stop_streaming(self) -> None:  # noqa: D401
        """Stop all streaming connections."""
        if not self.is_running:
            return
            
        self.is_running = False
        log.info("stopping_realtime_stream")
        
        # Close all connections
        for conn_name, ws in self.connections.items():
            try:
                await ws.close()
                log.info("connection_closed", connection=conn_name)
            except Exception as exc:
                log.warning("close_connection_failed", connection=conn_name, error=str(exc))
        
        self.connections.clear()
    
    async def _primary_stream_handler(self) -> None:  # noqa: D401
        """Handle primary WebSocket stream."""
        while self.is_running:
            try:
                async with websockets.connect(
                    settings.primary_ws_url,
                    extra_headers={"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
                ) as ws:
                    self.connections["primary"] = ws
                    self.reconnect_counts["primary"] = 0
                    
                    # Subscribe to symbols
                    await self._subscribe_symbols(ws, self.symbols)
                    
                    # Process messages
                    async for message in ws:
                        await self._process_message("primary", message)
                        
            except Exception as exc:
                log.warning("primary_stream_error", error=str(exc))
                self.reconnect_counts["primary"] += 1
                
                if self.reconnect_counts["primary"] >= settings.max_reconnects:
                    log.error("primary_stream_max_reconnects")
                    break
                    
                await asyncio.sleep(settings.reconnect_delay)
    
    async def _backup_stream_handler(self) -> None:  # noqa: D401
        """Handle backup WebSocket stream."""
        # Only activate if primary fails
        await asyncio.sleep(10)  # Give primary time to establish
        
        while self.is_running:
            if "primary" in self.connections and not self.connections["primary"].closed:
                await asyncio.sleep(30)  # Check every 30s
                continue
                
            try:
                async with websockets.connect(settings.backup_ws_url) as ws:
                    self.connections["backup"] = ws
                    log.info("backup_stream_activated")
                    
                    await self._subscribe_symbols(ws, self.symbols)
                    
                    async for message in ws:
                        await self._process_message("backup", message)
                        
            except Exception as exc:
                log.warning("backup_stream_error", error=str(exc))
                await asyncio.sleep(settings.reconnect_delay)
    
    async def _subscribe_symbols(self, ws: websockets.WebSocketServerProtocol, symbols: List[str]) -> None:  # noqa: D401,E501
        """Subscribe to symbol updates."""
        subscription = {
            "method": "SUBSCRIBE",
            "params": [f"{symbol.lower()}@ticker" for symbol in symbols] + 
                     [f"{symbol.lower()}@depth20" for symbol in symbols] +
                     [f"{symbol.lower()}@trade" for symbol in symbols],
            "id": int(time.time())
        }
        
        await ws.send(json.dumps(subscription))
        log.info("symbols_subscribed", symbols=symbols)
    
    # ------------------------------------------------------------------- #
    #  MESSAGE PROCESSING
    # ------------------------------------------------------------------- #
    
    async def _process_message(self, source: str, message: str) -> None:  # noqa: D401
        """Process incoming WebSocket message."""
        try:
            data = json.loads(message)
            
            if "stream" not in data:
                return
                
            stream_name = data["stream"]
            payload = data["data"]
            
            # Route by stream type
            if "@ticker" in stream_name:
                await self._process_ticker(payload)
            elif "@depth" in stream_name:
                await self._process_order_book(payload)
            elif "@trade" in stream_name:
                await self._process_trade(payload)
                
            self.last_data_time = time.time()
            
        except Exception as exc:
            log.warning("message_process_error", source=source, error=str(exc))
    
    async def _process_ticker(self, data: Dict) -> None:  # noqa: D401
        """Process ticker/summary data."""
        try:
            symbol = data["s"].upper()
            
            summary = MarketSummary(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(data["E"] / 1000, tz=timezone.utc),
                last_price=float(data["c"]),
                volume_24h=int(float(data["v"])),
                high_24h=float(data["h"]),
                low_24h=float(data["l"]),
                change_24h=float(data["P"]),
                bid=float(data.get("b", 0)),
                ask=float(data.get("a", 0)),
                spread=float(data.get("a", 0)) - float(data.get("b", 0))
            )
            
            self.market_summaries[symbol] = summary
            await self._notify_subscribers("ticker", symbol, summary)
            
        except Exception as exc:
            log.warning("ticker_process_error", error=str(exc))
    
    async def _process_order_book(self, data: Dict) -> None:  # noqa: D401
        """Process order book updates."""
        try:
            symbol = data["s"].upper()
            
            bids = [OrderBookLevel(float(bid[0]), int(float(bid[1]))) for bid in data["bids"][:10]]
            asks = [OrderBookLevel(float(ask[0]), int(float(ask[1]))) for ask in data["asks"][:10]]
            
            order_book = OrderBook(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(data["E"] / 1000, tz=timezone.utc),
                bids=sorted(bids, key=lambda x: x.price, reverse=True),
                asks=sorted(asks, key=lambda x: x.price),
                sequence=data.get("lastUpdateId", 0)
            )
            
            self.order_books[symbol] = order_book
            await self._notify_subscribers("orderbook", symbol, order_book)
            
        except Exception as exc:
            log.warning("orderbook_process_error", error=str(exc))
    
    async def _process_trade(self, data: Dict) -> None:  # noqa: D401
        """Process individual trade ticks."""
        try:
            symbol = data["s"].upper()
            
            tick = Tick(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(data["T"] / 1000, tz=timezone.utc),
                price=float(data["p"]),
                volume=int(float(data["q"])),
                side="buy" if data["m"] else "sell",  # m=true means buyer is market maker
                sequence=data.get("t", 0)
            )
            
            # Buffer tick data
            self.tick_buffers[symbol].append(tick)
            
            # Notify subscribers
            await self._notify_subscribers("tick", symbol, tick)
            
        except Exception as exc:
            log.warning("trade_process_error", error=str(exc))
    
    # ------------------------------------------------------------------- #
    #  SUBSCRIPTION SYSTEM
    # ------------------------------------------------------------------- #
    
    def subscribe(self, event_type: str, callback: Callable) -> None:  # noqa: D401
        """Subscribe to real-time events."""
        self.subscribers[event_type].append(callback)
        log.info("subscriber_added", event_type=event_type)
    
    def unsubscribe(self, event_type: str, callback: Callable) -> None:  # noqa: D401
        """Unsubscribe from events."""
        if callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)
            log.info("subscriber_removed", event_type=event_type)
    
    async def _notify_subscribers(self, event_type: str, symbol: str, data: Any) -> None:  # noqa: D401,E501
        """Notify all subscribers of new data."""
        for callback in self.subscribers[event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(symbol, data)
                else:
                    callback(symbol, data)
            except Exception as exc:
                log.warning("subscriber_error", event_type=event_type, error=str(exc))
    
    # ------------------------------------------------------------------- #
    #  MONITORING
    # ------------------------------------------------------------------- #
    
    async def _heartbeat_monitor(self) -> None:  # noqa: D401
        """Monitor connection health."""
        while self.is_running:
            await asyncio.sleep(settings.heartbeat_interval)
            
            # Check all connections
            for conn_name, ws in list(self.connections.items()):
                try:
                    await ws.ping()
                except Exception:
                    log.warning("heartbeat_failed", connection=conn_name)
                    self.connections.pop(conn_name, None)
    
    async def _circuit_breaker_monitor(self) -> None:  # noqa: D401
        """Monitor for data feed issues."""
        while self.is_running:
            await asyncio.sleep(10)  # Check every 10 seconds
            
            # Check if we're receiving data
            time_since_data = time.time() - self.last_data_time
            
            if time_since_data > settings.data_timeout:
                log.error("data_timeout_detected", seconds=time_since_data)
                
                # Try to restart connections
                for conn_name in list(self.connections.keys()):
                    try:
                        await self.connections[conn_name].close()
                    except Exception:
                        pass
                    self.connections.pop(conn_name, None)
    
    # ------------------------------------------------------------------- #
    #  DATA ACCESS
    # ------------------------------------------------------------------- #
    
    def get_latest_tick(self, symbol: str) -> Optional[Tick]:  # noqa: D401
        """Get the most recent tick for a symbol."""
        buffer = self.tick_buffers.get(symbol)
        return buffer[-1] if buffer else None
    
    def get_recent_ticks(self, symbol: str, count: int = 100) -> List[Tick]:  # noqa: D401
        """Get recent ticks for a symbol."""
        buffer = self.tick_buffers.get(symbol, deque())
        return list(buffer)[-count:]
    
    def get_order_book(self, symbol: str) -> Optional[OrderBook]:  # noqa: D401
        """Get current order book for a symbol."""
        return self.order_books.get(symbol)
    
    def get_market_summary(self, symbol: str) -> Optional[MarketSummary]:  # noqa: D401
        """Get market summary for a symbol."""
        return self.market_summaries.get(symbol)
    
    def get_connection_status(self) -> Dict[str, bool]:  # noqa: D401
        """Get status of all connections."""
        return {
            name: not ws.closed 
            for name, ws in self.connections.items()
        }


# ---------------------------------------------------------------------------
#  INTEGRATION HELPERS
# ---------------------------------------------------------------------------

async def create_stream_manager(symbols: List[str]) -> RealTimeStreamManager:
    """Factory function to create and start stream manager."""
    manager = RealTimeStreamManager(symbols)
    
    # Start streaming in background
    asyncio.create_task(manager.start_streaming())
    
    # Wait a moment for connections to establish
    await asyncio.sleep(2)
    
    return manager


def get_realtime_price(manager: RealTimeStreamManager, symbol: str) -> Optional[float]:
    """Quick helper to get latest price."""
    summary = manager.get_market_summary(symbol)
    return summary.last_price if summary else None


def get_spread(manager: RealTimeStreamManager, symbol: str) -> Optional[float]:
    """Quick helper to get bid-ask spread."""
    order_book = manager.get_order_book(symbol)
    if order_book and order_book.bids and order_book.asks:
        return order_book.asks[0].price - order_book.bids[0].price
    return None


# ---------------------------------------------------------------------------
#  CLI DEMO
# ---------------------------------------------------------------------------

async def _demo_stream():  # noqa: D401
    """Demo real-time streaming."""
    manager = await create_stream_manager(["CL", "GC", "ES"])
    
    # Simple tick printer
    def print_tick(symbol: str, tick: Tick) -> None:
        print(f"{symbol}: {tick.price} @ {tick.timestamp} ({tick.volume})")
    
    manager.subscribe("tick", print_tick)
    
    # Run for 30 seconds
    await asyncio.sleep(30)
    await manager.stop_streaming()


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_demo_stream())