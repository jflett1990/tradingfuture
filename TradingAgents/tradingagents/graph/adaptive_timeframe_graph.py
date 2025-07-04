"""adaptive_timeframe_graph.py – Multi-timeframe adaptive trading system.

Intelligently switches between trading styles based on market conditions:
• Scalping mode: 1-5 minute charts for range-bound, high-volume periods
• Day trading mode: 5-15 minute charts for trending intraday moves  
• Swing trading mode: 1-4 hour charts for longer-term directional moves

Features:
• Automatic timeframe detection based on volatility and volume
• Claude 4 analysis adapted to timeframe context
• Risk management scaled by holding period
• Strategy selection optimized for timeframe

© 2025 Prompt Maestro 9000, MIT License
"""
from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum

import structlog
import pandas as pd
import numpy as np
import yfinance as yf

from ..strategies.scalping_strategies import create_scalping_engine, TF as ScalpTF
from ..strategies.advanced_strategies import StrategyEngine
from .ib_enhanced_futures_graph import IBEnhancedFuturesGraph, create_ib_enhanced_graph
from ..default_config import DEFAULT_CONFIG

log = structlog.get_logger("adaptive_timeframe")

# ---------------------------------------------------------------------------
#  MARKET REGIME DETECTION
# ---------------------------------------------------------------------------

class MarketRegime(str, Enum):  # noqa: D101
    SCALPING = "scalping"          # Range-bound, high frequency
    DAY_TRADING = "day_trading"    # Intraday trending
    SWING_TRADING = "swing_trading" # Multi-day trends
    CHOPPY = "choppy"              # Difficult conditions

class TimeframeDetector:
    """Detects optimal timeframe based on market conditions."""
    
    def __init__(self):
        self.regime_history: Dict[str, List[MarketRegime]] = {}
    
    async def detect_regime(self, symbol: str, live_price: Optional[float] = None) -> Tuple[MarketRegime, Dict]:
        """Detect current market regime for symbol."""
        try:
            # Get multiple timeframes of data
            data_1m = yf.download(symbol, period="1d", interval="1m", progress=False)
            data_5m = yf.download(symbol, period="5d", interval="5m", progress=False)
            data_1h = yf.download(symbol, period="30d", interval="1h", progress=False)
            
            if len(data_1m) < 100 or len(data_5m) < 100 or len(data_1h) < 100:
                return MarketRegime.SWING_TRADING, {"reason": "insufficient_data"}
            
            # Calculate regime indicators
            metrics = {}
            
            # 1. Volatility analysis across timeframes
            returns_1m = data_1m['Close'].pct_change().dropna()
            returns_5m = data_5m['Close'].pct_change().dropna()
            returns_1h = data_1h['Close'].pct_change().dropna()
            
            vol_1m = returns_1m.std() * np.sqrt(1440)  # Annualized
            vol_5m = returns_5m.std() * np.sqrt(288)   # Annualized  
            vol_1h = returns_1h.std() * np.sqrt(24)    # Annualized
            
            metrics["volatility"] = {
                "1m": vol_1m,
                "5m": vol_5m, 
                "1h": vol_1h,
                "vol_ratio_1m_1h": vol_1m / vol_1h if vol_1h > 0 else 1.0
            }
            
            # 2. Volume analysis
            vol_1m_avg = data_1m['Volume'].rolling(20).mean().iloc[-1]
            vol_5m_avg = data_5m['Volume'].rolling(20).mean().iloc[-1] 
            vol_current = data_1m['Volume'].iloc[-10:].mean()
            
            volume_surge = vol_current / vol_1m_avg if vol_1m_avg > 0 else 1.0
            
            metrics["volume"] = {
                "surge": volume_surge,
                "consistency": data_1m['Volume'].rolling(20).std().iloc[-1] / vol_1m_avg if vol_1m_avg > 0 else 1.0
            }
            
            # 3. Trend strength analysis
            price_1h = data_1h['Close'].iloc[-1]
            price_1h_20 = data_1h['Close'].rolling(20).mean().iloc[-1]
            price_5m = data_5m['Close'].iloc[-1]
            price_5m_20 = data_5m['Close'].rolling(20).mean().iloc[-1]
            
            trend_1h = abs(price_1h - price_1h_20) / price_1h_20 if price_1h_20 > 0 else 0
            trend_5m = abs(price_5m - price_5m_20) / price_5m_20 if price_5m_20 > 0 else 0
            
            metrics["trend"] = {
                "strength_1h": trend_1h,
                "strength_5m": trend_5m,
                "alignment": 1.0 if np.sign(price_1h - price_1h_20) == np.sign(price_5m - price_5m_20) else 0.0
            }
            
            # 4. Range analysis (for scalping detection)
            high_1h = data_1h['High'].rolling(24).max().iloc[-1]  # 24h high
            low_1h = data_1h['Low'].rolling(24).min().iloc[-1]    # 24h low
            current_price = live_price or data_1m['Close'].iloc[-1]
            
            range_position = (current_price - low_1h) / (high_1h - low_1h) if high_1h > low_1h else 0.5
            range_size = (high_1h - low_1h) / current_price if current_price > 0 else 0
            
            metrics["range"] = {
                "position": range_position,  # 0-1, where price is in daily range
                "size": range_size,          # Range as % of price
                "middle_zone": 0.3 < range_position < 0.7  # In middle of range
            }
            
            # 5. Market microstructure (for scalping)
            if len(data_1m) >= 60:
                price_changes = data_1m['Close'].diff().iloc[-60:]  # Last hour
                mean_reversion = np.corrcoef(range(len(price_changes)), price_changes)[0,1]  # Correlation with time
                mean_reversion = abs(mean_reversion) if not np.isnan(mean_reversion) else 0
                
                metrics["microstructure"] = {
                    "mean_reversion": mean_reversion,
                    "small_moves": np.sum(np.abs(price_changes) < current_price * 0.001) / len(price_changes)
                }
            else:
                metrics["microstructure"] = {"mean_reversion": 0, "small_moves": 0}
            
            # Decision logic
            regime = self._classify_regime(metrics)
            
            # Store in history for stability
            if symbol not in self.regime_history:
                self.regime_history[symbol] = []
            self.regime_history[symbol].append(regime)
            
            # Keep last 10 classifications
            if len(self.regime_history[symbol]) > 10:
                self.regime_history[symbol] = self.regime_history[symbol][-10:]
            
            # Smooth regime changes (require 3/5 agreement)
            if len(self.regime_history[symbol]) >= 5:
                recent_regimes = self.regime_history[symbol][-5:]
                regime_counts = {r: recent_regimes.count(r) for r in set(recent_regimes)}
                stable_regime = max(regime_counts.keys(), key=lambda r: regime_counts[r])
                
                if regime_counts[stable_regime] >= 3:
                    regime = stable_regime
            
            log.info("regime_detected", 
                    symbol=symbol, 
                    regime=regime,
                    volatility=vol_1m,
                    volume_surge=volume_surge,
                    trend_strength=trend_1h)
            
            return regime, metrics
            
        except Exception as exc:
            log.error("regime_detection_error", symbol=symbol, error=str(exc))
            return MarketRegime.SWING_TRADING, {"error": str(exc)}
    
    def _classify_regime(self, metrics: Dict) -> MarketRegime:
        """Classify market regime based on metrics."""
        vol = metrics["volatility"]
        volume = metrics["volume"]
        trend = metrics["trend"] 
        range_data = metrics["range"]
        micro = metrics["microstructure"]
        
        # Scalping conditions
        scalping_score = 0
        
        # High volume activity
        if volume["surge"] > 2.0:
            scalping_score += 2
        
        # Range-bound market
        if range_data["middle_zone"] and range_data["size"] < 0.02:  # < 2% daily range
            scalping_score += 2
            
        # Mean reversion characteristics  
        if micro["mean_reversion"] < 0.3 and micro["small_moves"] > 0.7:
            scalping_score += 2
            
        # Low trend on short timeframes
        if trend["strength_5m"] < 0.01:
            scalping_score += 1
        
        # Day trading conditions
        day_trading_score = 0
        
        # Moderate trend with volume
        if 0.01 < trend["strength_5m"] < 0.05 and volume["surge"] > 1.5:
            day_trading_score += 3
            
        # Trend alignment between timeframes
        if trend["alignment"] > 0.5 and trend["strength_1h"] > 0.02:
            day_trading_score += 2
            
        # Good volatility for intraday
        if 0.3 < vol["vol_ratio_1m_1h"] < 1.5:
            day_trading_score += 1
        
        # Swing trading conditions (default)
        swing_trading_score = 0
        
        # Strong longer-term trend
        if trend["strength_1h"] > 0.05:
            swing_trading_score += 3
            
        # Lower frequency volatility
        if vol["vol_ratio_1m_1h"] < 0.8:
            swing_trading_score += 2
            
        # Determine regime
        scores = {
            MarketRegime.SCALPING: scalping_score,
            MarketRegime.DAY_TRADING: day_trading_score, 
            MarketRegime.SWING_TRADING: swing_trading_score + 1  # Slight bias toward swing
        }
        
        max_score = max(scores.values())
        if max_score < 3:
            return MarketRegime.CHOPPY
            
        return max(scores.keys(), key=lambda r: scores[r])

# ---------------------------------------------------------------------------
#  ADAPTIVE TRADING SYSTEM
# ---------------------------------------------------------------------------

class AdaptiveTimeframeGraph:
    """Multi-timeframe adaptive trading system."""
    
    def __init__(self, debug: bool = False, config: Optional[Dict] = None, paper_trading: bool = True):
        self.debug = debug
        self.config = config or DEFAULT_CONFIG.copy()
        self.paper_trading = paper_trading
        
        # Components
        self.detector = TimeframeDetector()
        self.scalping_engine = None
        self.swing_graph = None
        
        # State tracking
        self.current_regime: Dict[str, MarketRegime] = {}
        self.regime_confidence: Dict[str, float] = {}
    
    async def analyze_adaptive(self, symbol: str, trade_date: str) -> Dict:
        """Run adaptive analysis based on detected market regime."""
        
        # Detect current market regime
        regime, regime_metrics = await self.detector.detect_regime(symbol)
        self.current_regime[symbol] = regime
        
        log.info("adaptive_analysis_start", 
                symbol=symbol, 
                regime=regime,
                date=trade_date)
        
        try:
            if regime == MarketRegime.SCALPING:
                return await self._scalping_analysis(symbol, regime_metrics)
            elif regime == MarketRegime.DAY_TRADING:
                return await self._day_trading_analysis(symbol, regime_metrics)
            elif regime == MarketRegime.SWING_TRADING:
                return await self._swing_analysis(symbol, trade_date, regime_metrics)
            else:  # CHOPPY
                return await self._choppy_analysis(symbol, regime_metrics)
                
        except Exception as exc:
            log.error("adaptive_analysis_error", symbol=symbol, regime=regime, error=str(exc))
            # Fallback to swing analysis
            return await self._swing_analysis(symbol, trade_date, regime_metrics)
    
    async def _scalping_analysis(self, symbol: str, metrics: Dict) -> Dict:
        """Scalping-optimized analysis."""
        if not self.scalping_engine:
            self.scalping_engine = await create_scalping_engine([symbol], ScalpTF.M1)
        
        # Get minute-level data for scalping
        data = yf.download(symbol, period="1d", interval="1m", progress=False)
        
        # Create mock tick data from minute bars
        tick_data = []
        for i, (timestamp, row) in enumerate(data.iterrows()):
            tick_data.append({
                'price': row['Close'],
                'volume': row['Volume'] // 60,  # Simulate ticks
                'timestamp': timestamp.tz_localize('UTC') if timestamp.tz is None else timestamp
            })
        
        # Mock order book (in real implementation, this comes from IB)
        order_book = {
            'bids': [{'price': data['Close'].iloc[-1] - 0.01 * i, 'size': 100} for i in range(1, 6)],
            'asks': [{'price': data['Close'].iloc[-1] + 0.01 * i, 'size': 100} for i in range(1, 6)]
        }
        
        # Run scalping analysis
        signal = await self.scalping_engine.scalp_analysis(symbol, tick_data, order_book, data)
        
        if signal:
            decision = {
                "symbol": symbol,
                "trading_style": "scalping",
                "timeframe": "1m",
                "action": signal.action,
                "confidence": signal.confidence,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "position_size": signal.position_size,
                "strategy": signal.strategy,
                "reasoning": f"SCALPING MODE: {signal.reasoning}",
                "hold_time_target": "seconds_to_minutes",
                "regime_metrics": metrics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        else:
            decision = {
                "symbol": symbol,
                "trading_style": "scalping", 
                "action": "HOLD",
                "reasoning": "No scalping opportunities detected",
                "regime_metrics": metrics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        return decision
    
    async def _day_trading_analysis(self, symbol: str, metrics: Dict) -> Dict:
        """Day trading optimized analysis."""
        # Use 5-minute data for day trading
        data = yf.download(symbol, period="5d", interval="5m", progress=False)
        
        # Initialize strategy engine for day trading
        strategy_engine = StrategyEngine([symbol])
        
        try:
            # Run strategies optimized for day trading timeframe
            signal = await strategy_engine.multi_signal(data, symbol)
            
            decision = {
                "symbol": symbol,
                "trading_style": "day_trading",
                "timeframe": "5m", 
                "action": signal.action if signal else "HOLD",
                "confidence": signal.confidence if signal else 0.5,
                "entry_price": signal.entry_price if signal else data['Close'].iloc[-1],
                "stop_loss": signal.stop_loss if signal else 0,
                "take_profit": signal.take_profit if signal else 0,
                "position_size": signal.position_size if signal else 0,
                "strategy": signal.strategy if signal else "day_trading_hold",
                "reasoning": f"DAY TRADING MODE: {signal.reasoning if signal else 'No intraday opportunities'}",
                "hold_time_target": "minutes_to_hours",
                "regime_metrics": metrics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        finally:
            await strategy_engine.close()
        
        return decision
    
    async def _swing_analysis(self, symbol: str, trade_date: str, metrics: Dict) -> Dict:
        """Swing trading analysis using enhanced graph."""
        if not self.swing_graph:
            self.swing_graph = await create_ib_enhanced_graph(
                paper_trading=self.paper_trading,
                config=self.config,
                debug=self.debug
            )
        
        # Run full enhanced analysis
        decision = await self.swing_graph.propagate(symbol, trade_date)
        
        # Add adaptive context
        decision.update({
            "trading_style": "swing_trading",
            "timeframe": "1h",
            "hold_time_target": "hours_to_days", 
            "regime_metrics": metrics,
            "reasoning": f"SWING MODE: {decision.get('reasoning', '')}"
        })
        
        return decision
    
    async def _choppy_analysis(self, symbol: str, metrics: Dict) -> Dict:
        """Analysis for choppy/difficult market conditions."""
        return {
            "symbol": symbol,
            "trading_style": "choppy_avoidance",
            "action": "HOLD",
            "confidence": 0.0,
            "reasoning": "CHOPPY MARKET: Difficult conditions detected, avoiding trades",
            "hold_time_target": "wait",
            "regime_metrics": metrics,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.scalping_engine:
            await self.scalping_engine.close()
        if self.swing_graph:
            await self.swing_graph.cleanup()

# ---------------------------------------------------------------------------
#  FACTORY AND DEMO
# ---------------------------------------------------------------------------

async def create_adaptive_graph(paper_trading: bool = True, config: Optional[Dict] = None, 
                               debug: bool = False) -> AdaptiveTimeframeGraph:
    """Create adaptive timeframe trading graph."""
    graph = AdaptiveTimeframeGraph(debug=debug, config=config, paper_trading=paper_trading)
    
    log.info("adaptive_graph_created", 
            paper_trading=paper_trading,
            timeframes=["scalping", "day_trading", "swing_trading"])
    
    return graph

async def _demo_adaptive():
    """Demo adaptive timeframe detection."""
    symbols = ["ES", "CL", "GC"]
    
    graph = await create_adaptive_graph(paper_trading=True, debug=True)
    
    print("🔄 ADAPTIVE TIMEFRAME DEMO")
    print("="*50)
    
    for symbol in symbols:
        print(f"\n📊 Analyzing {symbol}...")
        
        decision = await graph.analyze_adaptive(symbol, "2025-01-15")
        
        print(f"Symbol: {decision['symbol']}")
        print(f"Detected Style: {decision['trading_style']}")
        print(f"Timeframe: {decision.get('timeframe', 'N/A')}")
        print(f"Action: {decision['action']}")
        print(f"Confidence: {decision.get('confidence', 0):.1%}")
        print(f"Hold Target: {decision.get('hold_time_target', 'N/A')}")
        print(f"Reasoning: {decision['reasoning']}")
        
        # Show regime metrics
        regime_metrics = decision.get('regime_metrics', {})
        if 'volatility' in regime_metrics:
            vol = regime_metrics['volatility']
            print(f"Volatility 1m/1h ratio: {vol.get('vol_ratio_1m_1h', 0):.2f}")
        
        if 'volume' in regime_metrics:
            vol_data = regime_metrics['volume']
            print(f"Volume surge: {vol_data.get('surge', 0):.1f}x")
    
    await graph.cleanup()
    print("\n✅ Demo complete!")

if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_demo_adaptive())