# Advanced Futures Trading Strategies with Real-time News Integration

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

from ..dataflows.news_sentinel import NewsSentinel, Snapshot


@dataclass
class TradeSignal:
    symbol: str
    action: str  # BUY, SELL, HOLD, CLOSE
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    strategy_name: str
    reasoning: str
    risk_reward_ratio: float
    urgency: str  # low, medium, high
    news_catalyst: bool
    timestamp: datetime


@dataclass
class MarketRegime:
    regime_type: str  # trending, mean_reverting, volatile, quiet, news_driven
    strength: float
    duration: int  # days
    characteristics: Dict[str, float]
    dominant_factor: str  # technical, fundamental, sentiment


class AdvancedFuturesStrategies:
    def __init__(self, config: Dict):
        self.config = config
        self.lookback_periods = {
            'ultra_short': 5,
            'short': 20,
            'medium': 50,
            'long': 200
        }
        self.regime_cache = {}
        self.news_sentinel = None
        self.position_cache = {}
        
    async def initialize_news_engine(self, symbols: List[str]):
        """Initialize the news sentiment engine"""
        self.news_sentinel = NewsSentinel(symbols)
        
    async def get_news_sentiment(self, symbol: str) -> Optional[Snapshot]:
        """Get real-time news sentiment for symbol"""
        if not self.news_sentinel:
            return None
        try:
            return await self.news_sentinel.snapshot(symbol)
        except Exception as e:
            print(f"Error getting news sentiment for {symbol}: {e}")
            return None
    
    def detect_market_regime(self, data: pd.DataFrame, news_data: Optional[Snapshot] = None) -> MarketRegime:
        """Enhanced regime detection with news sentiment integration"""
        try:
            returns = data['Close'].pct_change().dropna()
            
            # Technical regime indicators
            high_low = data['High'] - data['Low']
            high_close = abs(data['High'] - data['Close'].shift(1))
            low_close = abs(data['Low'] - data['Close'].shift(1))
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            
            # ADX calculation
            plus_dm = np.where((data['High'] - data['High'].shift(1)) > 
                              (data['Low'].shift(1) - data['Low']),
                              np.maximum(data['High'] - data['High'].shift(1), 0), 0)
            minus_dm = np.where((data['Low'].shift(1) - data['Low']) > 
                               (data['High'] - data['High'].shift(1)),
                               np.maximum(data['Low'].shift(1) - data['Low'], 0), 0)
            
            atr = pd.Series(true_range).rolling(14).mean()
            plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / atr)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(14).mean()
            
            # Volatility and momentum
            volatility_20 = returns.rolling(20).std() * np.sqrt(252)
            momentum_20 = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1) * 100
            
            # Volume analysis
            volume_ratio = data['Volume'].iloc[-10:].mean() / data['Volume'].rolling(50).mean().iloc[-1]
            
            # News sentiment integration
            news_factor = 0.0
            news_urgency = 0.0
            news_confidence = 0.0
            
            if news_data:
                news_factor = abs(news_data.score)
                news_urgency = max([item.urgency for item in news_data.news_items] + [0])
                news_confidence = news_data.confidence
            
            # Current values
            current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 20
            current_vol = volatility_20.iloc[-1] if not pd.isna(volatility_20.iloc[-1]) else 0.2
            
            characteristics = {
                'trend_strength': current_adx,
                'volatility': current_vol,
                'momentum': momentum_20,
                'volume_ratio': volume_ratio,
                'news_factor': news_factor,
                'news_urgency': news_urgency,
                'news_confidence': news_confidence
            }
            
            # Enhanced regime classification with news
            regime_type = 'quiet'
            strength = 0.5
            dominant_factor = 'technical'
            
            # News-driven regime (highest priority)
            if news_urgency > 0.6 and news_confidence > 0.5:
                regime_type = 'news_driven'
                strength = min(1.0, news_urgency + news_factor * 0.5)
                dominant_factor = 'sentiment'
            
            # Technical regimes
            elif current_adx > 30 and current_vol < 0.3:
                regime_type = 'trending'
                strength = min(1.0, current_adx / 50)
                dominant_factor = 'technical'
            
            elif current_vol > 0.5 or volume_ratio > 2.0:
                regime_type = 'volatile'
                strength = min(1.0, current_vol / 0.8)
                dominant_factor = 'technical'
            
            elif abs(momentum_20) < 5 and current_vol < 0.25:
                regime_type = 'mean_reverting'
                strength = min(1.0, (0.25 - current_vol) / 0.25)
                dominant_factor = 'technical'
            
            return MarketRegime(
                regime_type=regime_type,
                strength=strength,
                duration=self._estimate_regime_duration(data),
                characteristics=characteristics,
                dominant_factor=dominant_factor
            )
            
        except Exception as e:
            print(f"Error detecting market regime: {e}")
            return MarketRegime('quiet', 0.5, 10, {}, 'technical')
    
    def _estimate_regime_duration(self, data: pd.DataFrame) -> int:
        """Estimate regime duration"""
        try:
            returns = data['Close'].pct_change().dropna()
            volatility = returns.rolling(20).std()
            vol_changes = volatility.rolling(5).mean().pct_change().abs()
            recent_changes = vol_changes.iloc[-20:]
            
            threshold = recent_changes.quantile(0.8)
            change_points = recent_changes[recent_changes > threshold]
            
            if len(change_points) > 0:
                last_change = change_points.index[-1]
                days_since_change = len(data) - data.index.get_loc(last_change)
                return min(days_since_change, 50)
            
            return 20
        except:
            return 20
    
    async def lightning_news_strategy(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Ultra-fast news-driven strategy for immediate market reactions"""
        news_data = await self.get_news_sentiment(symbol)
        
        if not news_data or not news_data.news_items:
            return self._default_signal(data, symbol, "No recent news")
        
        # Get the most urgent recent news
        recent_news = [item for item in news_data.news_items 
                      if (datetime.now() - item.timestamp.replace(tzinfo=None)).total_seconds() < 3600]
        
        if not recent_news:
            return self._default_signal(data, symbol, "No recent urgent news")
        
        # Calculate news impact
        max_urgency = max(item.urgency for item in recent_news)
        sentiment_strength = abs(news_data.score)
        
        # Only trade on high-impact news
        if max_urgency < 0.5 or sentiment_strength < 0.2:
            return self._default_signal(data, symbol, "News impact too low")
        
        current_price = data['Close'].iloc[-1]
        atr = talib.ATR(data['High'].values, data['Low'].values, data['Close'].values, timeperiod=14)[-1]
        
        # Calculate position based on news intensity
        news_multiplier = max_urgency * sentiment_strength * news_data.confidence
        confidence = min(0.95, 0.6 + news_multiplier * 0.3)
        
        # Determine direction
        if news_data.label == 'bullish':
            action = 'BUY'
            stop_loss = current_price - (atr * 1.5)
            take_profit = current_price + (atr * 3.0)
            reasoning = f"URGENT: Bullish news catalyst - urgency: {max_urgency:.2f}, sentiment: {sentiment_strength:.2f}"
        
        elif news_data.label == 'bearish':
            action = 'SELL'
            stop_loss = current_price + (atr * 1.5)
            take_profit = current_price - (atr * 3.0)
            reasoning = f"URGENT: Bearish news catalyst - urgency: {max_urgency:.2f}, sentiment: {sentiment_strength:.2f}"
        
        else:
            return self._default_signal(data, symbol, "Neutral sentiment")
        
        return TradeSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=self._calculate_position_size(atr, None) * news_multiplier,
            strategy_name='lightning_news',
            reasoning=reasoning,
            risk_reward_ratio=abs(take_profit - current_price) / abs(current_price - stop_loss),
            urgency='high',
            news_catalyst=True,
            timestamp=datetime.now()
        )
    
    async def adaptive_momentum_strategy(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Momentum strategy that adapts to market regime and news"""
        news_data = await self.get_news_sentiment(symbol)
        regime = self.detect_market_regime(data, news_data)
        
        if regime.regime_type not in ['trending', 'news_driven']:
            return self._default_signal(data, symbol, f"Regime not suitable for momentum: {regime.regime_type}")
        
        closes = data['Close']
        highs = data['High']
        lows = data['Low']
        volumes = data['Volume']
        
        # Adaptive lookback based on regime
        if regime.regime_type == 'news_driven':
            lookback = self.lookback_periods['ultra_short']
        elif regime.strength > 0.8:
            lookback = self.lookback_periods['short']
        else:
            lookback = self.lookback_periods['medium']
        
        # Technical indicators
        rsi = talib.RSI(closes.values, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(closes.values)
        atr = talib.ATR(highs.values, lows.values, closes.values, timeperiod=14)
        
        # Bollinger Bands with regime adjustment
        bb_period = max(10, lookback)
        bb_std = 2.0 + (regime.characteristics.get('volatility', 0.2) * 1.5)
        bb_middle = closes.rolling(bb_period).mean()
        bb_std_dev = closes.rolling(bb_period).std()
        bb_upper = bb_middle + (bb_std_dev * bb_std)
        bb_lower = bb_middle - (bb_std_dev * bb_std)
        
        current_price = closes.iloc[-1]
        current_rsi = rsi[-1]
        current_atr = atr[-1]
        
        # Volume confirmation
        volume_ma = volumes.rolling(20).mean()
        volume_spike = volumes.iloc[-1] > (volume_ma.iloc[-1] * 1.3)
        
        # Momentum signals
        momentum_5 = (current_price / closes.iloc[-5] - 1) * 100
        momentum_20 = (current_price / closes.iloc[-20] - 1) * 100
        
        # News boost
        news_boost = 0
        if news_data and regime.regime_type == 'news_driven':
            if news_data.label == 'bullish' and momentum_5 > 0:
                news_boost = 0.3
            elif news_data.label == 'bearish' and momentum_5 < 0:
                news_boost = 0.3
        
        # Entry conditions
        bullish_momentum = (
            current_price > bb_upper.iloc[-1] and
            momentum_5 > 0.5 and
            current_rsi < 75 and
            macd[-1] > macd_signal[-1] and
            volume_spike
        )
        
        bearish_momentum = (
            current_price < bb_lower.iloc[-1] and
            momentum_5 < -0.5 and
            current_rsi > 25 and
            macd[-1] < macd_signal[-1] and
            volume_spike
        )
        
        if bullish_momentum:
            confidence = min(0.9, 0.5 + abs(momentum_5) * 0.02 + news_boost)
            stop_loss = max(bb_middle.iloc[-1], current_price - (current_atr * 2))
            take_profit = current_price + (current_atr * 3)
            
            return TradeSignal(
                symbol=symbol,
                action='BUY',
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=self._calculate_position_size(current_atr, regime),
                strategy_name='adaptive_momentum',
                reasoning=f"Bullish momentum: {momentum_5:.1f}%, regime: {regime.regime_type}",
                risk_reward_ratio=(take_profit - current_price) / (current_price - stop_loss),
                urgency='medium' if regime.regime_type == 'news_driven' else 'low',
                news_catalyst=regime.regime_type == 'news_driven',
                timestamp=datetime.now()
            )
        
        elif bearish_momentum:
            confidence = min(0.9, 0.5 + abs(momentum_5) * 0.02 + news_boost)
            stop_loss = min(bb_middle.iloc[-1], current_price + (current_atr * 2))
            take_profit = current_price - (current_atr * 3)
            
            return TradeSignal(
                symbol=symbol,
                action='SELL',
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=self._calculate_position_size(current_atr, regime),
                strategy_name='adaptive_momentum',
                reasoning=f"Bearish momentum: {momentum_5:.1f}%, regime: {regime.regime_type}",
                risk_reward_ratio=(current_price - take_profit) / (stop_loss - current_price),
                urgency='medium' if regime.regime_type == 'news_driven' else 'low',
                news_catalyst=regime.regime_type == 'news_driven',
                timestamp=datetime.now()
            )
        
        return self._default_signal(data, symbol, "No momentum signal detected")
    
    async def volatility_expansion_strategy(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Strategy that capitalizes on volatility expansion with news confirmation"""
        news_data = await self.get_news_sentiment(symbol)
        
        closes = data['Close']
        highs = data['High']
        lows = data['Low']
        volumes = data['Volume']
        
        # ATR and volatility metrics
        atr_14 = talib.ATR(highs.values, lows.values, closes.values, timeperiod=14)
        atr_50 = talib.ATR(highs.values, lows.values, closes.values, timeperiod=50)
        
        current_atr = atr_14[-1]
        avg_atr = np.mean(atr_50[-20:])
        
        # Volatility percentile
        atr_percentile = stats.percentileofscore(atr_14[-100:], current_atr)
        
        # Bollinger Band squeeze
        bb_period = 20
        bb_upper = closes.rolling(bb_period).mean() + (closes.rolling(bb_period).std() * 2)
        bb_lower = closes.rolling(bb_period).mean() - (closes.rolling(bb_period).std() * 2)
        bb_width = (bb_upper - bb_lower) / closes.rolling(bb_period).mean()
        
        # Keltner Channels
        kc_middle = closes.rolling(bb_period).mean()
        kc_range = pd.Series(atr_14).rolling(bb_period).mean() * 2
        kc_upper = kc_middle + kc_range
        kc_lower = kc_middle - kc_range
        
        # Squeeze condition
        squeeze = (bb_width.iloc[-1] < bb_width.rolling(50).quantile(0.2).iloc[-1])
        
        current_price = closes.iloc[-1]
        
        # Volume expansion
        volume_ma = volumes.rolling(20).mean()
        volume_expansion = volumes.iloc[-1] > (volume_ma.iloc[-1] * 1.5)
        
        # News catalyst boost
        news_catalyst = False
        if news_data and news_data.news_items:
            recent_urgent_news = [item for item in news_data.news_items 
                                if item.urgency > 0.4 and 
                                (datetime.now() - item.timestamp.replace(tzinfo=None)).total_seconds() < 7200]
            news_catalyst = len(recent_urgent_news) > 0
        
        # Volatility expansion conditions
        vol_expansion = (
            current_atr > avg_atr * 1.3 and
            atr_percentile > 70 and
            volume_expansion
        )
        
        # Breakout direction
        upper_breakout = current_price > kc_upper.iloc[-1]
        lower_breakout = current_price < kc_lower.iloc[-1]
        
        if vol_expansion and (upper_breakout or lower_breakout):
            base_confidence = 0.6 + (atr_percentile - 70) * 0.01
            
            if news_catalyst:
                base_confidence += 0.2
                urgency = 'high'
            else:
                urgency = 'medium'
            
            if upper_breakout:
                stop_loss = kc_middle.iloc[-1]
                take_profit = current_price + (current_atr * 2.5)
                
                return TradeSignal(
                    symbol=symbol,
                    action='BUY',
                    confidence=min(0.9, base_confidence),
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=self._calculate_position_size(current_atr, None),
                    strategy_name='volatility_expansion',
                    reasoning=f"Vol expansion breakout: ATR {atr_percentile:.0f}th percentile",
                    risk_reward_ratio=(take_profit - current_price) / (current_price - stop_loss),
                    urgency=urgency,
                    news_catalyst=news_catalyst,
                    timestamp=datetime.now()
                )
            
            else:  # lower_breakout
                stop_loss = kc_middle.iloc[-1]
                take_profit = current_price - (current_atr * 2.5)
                
                return TradeSignal(
                    symbol=symbol,
                    action='SELL',
                    confidence=min(0.9, base_confidence),
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=self._calculate_position_size(current_atr, None),
                    strategy_name='volatility_expansion',
                    reasoning=f"Vol expansion breakdown: ATR {atr_percentile:.0f}th percentile",
                    risk_reward_ratio=(current_price - take_profit) / (stop_loss - current_price),
                    urgency=urgency,
                    news_catalyst=news_catalyst,
                    timestamp=datetime.now()
                )
        
        return self._default_signal(data, symbol, "No volatility expansion detected")
    
    async def mean_reversion_with_sentiment(self, data: pd.DataFrame, symbol: str) -> TradeSignal:
        """Mean reversion strategy enhanced with sentiment analysis"""
        news_data = await self.get_news_sentiment(symbol)
        regime = self.detect_market_regime(data, news_data)
        
        # Avoid mean reversion in strong trending or news-driven markets
        if regime.regime_type in ['trending', 'news_driven'] and regime.strength > 0.7:
            return self._default_signal(data, symbol, f"Strong {regime.regime_type} regime, avoiding mean reversion")
        
        closes = data['Close']
        highs = data['High']
        lows = data['Low']
        
        # Z-score and statistical measures
        lookback = 30
        mean_price = closes.rolling(lookback).mean()
        std_price = closes.rolling(lookback).std()
        z_score = (closes.iloc[-1] - mean_price.iloc[-1]) / std_price.iloc[-1]
        
        # RSI and other oscillators
        rsi = talib.RSI(closes.values, timeperiod=14)
        stoch_k, stoch_d = talib.STOCH(highs.values, lows.values, closes.values)
        
        current_price = closes.iloc[-1]
        current_rsi = rsi[-1]
        
        # ATR for position sizing
        atr = talib.ATR(highs.values, lows.values, closes.values, timeperiod=14)[-1]
        
        # News sentiment filter
        sentiment_filter = True
        if news_data:
            # Avoid mean reversion against strong news sentiment
            if abs(news_data.score) > 0.4 and news_data.confidence > 0.6:
                sentiment_filter = False
        
        # Mean reversion conditions
        oversold = (
            z_score < -2.0 and
            current_rsi < 25 and
            stoch_k[-1] < 20 and
            sentiment_filter
        )
        
        overbought = (
            z_score > 2.0 and
            current_rsi > 75 and
            stoch_k[-1] > 80 and
            sentiment_filter
        )
        
        if oversold:
            confidence = min(0.8, 0.4 + abs(z_score) * 0.1 + (25 - current_rsi) * 0.01)
            stop_loss = current_price - (atr * 1.5)
            take_profit = mean_price.iloc[-1]
            
            return TradeSignal(
                symbol=symbol,
                action='BUY',
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=self._calculate_position_size(atr, regime) * 0.7,
                strategy_name='mean_reversion_sentiment',
                reasoning=f"Oversold mean reversion: Z-score {z_score:.2f}, RSI {current_rsi:.1f}",
                risk_reward_ratio=(take_profit - current_price) / (current_price - stop_loss),
                urgency='low',
                news_catalyst=False,
                timestamp=datetime.now()
            )
        
        elif overbought:
            confidence = min(0.8, 0.4 + abs(z_score) * 0.1 + (current_rsi - 75) * 0.01)
            stop_loss = current_price + (atr * 1.5)
            take_profit = mean_price.iloc[-1]
            
            return TradeSignal(
                symbol=symbol,
                action='SELL',
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=self._calculate_position_size(atr, regime) * 0.7,
                strategy_name='mean_reversion_sentiment',
                reasoning=f"Overbought mean reversion: Z-score {z_score:.2f}, RSI {current_rsi:.1f}",
                risk_reward_ratio=(current_price - take_profit) / (stop_loss - current_price),
                urgency='low',
                news_catalyst=False,
                timestamp=datetime.now()
            )
        
        return self._default_signal(data, symbol, "No mean reversion opportunity")
    
    def _calculate_position_size(self, atr: float, regime: Optional[MarketRegime] = None) -> float:
        """Enhanced position sizing with regime and news consideration"""
        try:
            base_size = self.config.get('position_size_limit', 0.1)
            
            # ATR adjustment
            if atr > 0:
                volatility_adjustment = min(2.0, 1.0 / (atr / 10))
            else:
                volatility_adjustment = 1.0
            
            # Regime adjustment
            regime_adjustment = 1.0
            if regime:
                if regime.regime_type == 'volatile':
                    regime_adjustment = 0.6
                elif regime.regime_type == 'news_driven':
                    # Increase size for high-confidence news
                    news_confidence = regime.characteristics.get('news_confidence', 0.5)
                    regime_adjustment = 0.8 + (news_confidence * 0.5)
                elif regime.regime_type == 'trending' and regime.strength > 0.8:
                    regime_adjustment = 1.2
                elif regime.regime_type == 'mean_reverting':
                    regime_adjustment = 0.7
            
            return base_size * volatility_adjustment * regime_adjustment
        except:
            return 0.05
    
    def _default_signal(self, data: pd.DataFrame, symbol: str, reason: str = "No signal") -> TradeSignal:
        """Return default HOLD signal"""
        current_price = data['Close'].iloc[-1] if not data.empty else 0
        
        return TradeSignal(
            symbol=symbol,
            action='HOLD',
            confidence=0.5,
            entry_price=current_price,
            stop_loss=0,
            take_profit=0,
            position_size=0,
            strategy_name='default',
            reasoning=reason,
            risk_reward_ratio=0,
            urgency='low',
            news_catalyst=False,
            timestamp=datetime.now()
        )
    
    async def get_multi_strategy_signals(self, data: pd.DataFrame, symbol: str) -> Dict[str, TradeSignal]:
        """Get signals from all advanced strategies"""
        strategies = {
            'lightning_news': self.lightning_news_strategy,
            'adaptive_momentum': self.adaptive_momentum_strategy,
            'volatility_expansion': self.volatility_expansion_strategy,
            'mean_reversion_sentiment': self.mean_reversion_with_sentiment,
        }
        
        signals = {}
        for name, strategy_func in strategies.items():
            try:
                signal = await strategy_func(data, symbol)
                signals[name] = signal
            except Exception as e:
                print(f"Error in strategy {name}: {e}")
                signals[name] = self._default_signal(data, symbol, f"Error in {name}")
        
        return signals
    
    async def combine_signals_advanced(self, signals: Dict[str, TradeSignal]) -> TradeSignal:
        """Advanced signal combination with priority weighting"""
        try:
            # Priority weighting for strategies
            strategy_weights = {
                'lightning_news': 3.0,  # Highest priority for urgent news
                'adaptive_momentum': 2.0,
                'volatility_expansion': 1.5,
                'mean_reversion_sentiment': 1.0
            }
            
            # Filter active signals
            active_signals = {k: v for k, v in signals.items() if v.action in ['BUY', 'SELL']}
            
            if not active_signals:
                return list(signals.values())[0]  # Return first HOLD signal
            
            # Calculate weighted scores
            buy_score = 0
            sell_score = 0
            
            for name, signal in active_signals.items():
                weight = strategy_weights.get(name, 1.0)
                
                # Boost weight for urgent signals
                if signal.urgency == 'high':
                    weight *= 2.0
                elif signal.urgency == 'medium':
                    weight *= 1.5
                
                # Boost weight for news-driven signals
                if signal.news_catalyst:
                    weight *= 1.5
                
                weighted_confidence = signal.confidence * weight
                
                if signal.action == 'BUY':
                    buy_score += weighted_confidence
                else:
                    sell_score += weighted_confidence
            
            # Determine consensus
            if buy_score > sell_score * 1.1:  # Slight bias toward action
                consensus_action = 'BUY'
                consensus_confidence = buy_score / sum(strategy_weights[k] for k in active_signals.keys())
                reference_signal = next(s for s in active_signals.values() if s.action == 'BUY')
            elif sell_score > buy_score * 1.1:
                consensus_action = 'SELL'
                consensus_confidence = sell_score / sum(strategy_weights[k] for k in active_signals.keys())
                reference_signal = next(s for s in active_signals.values() if s.action == 'SELL')
            else:
                consensus_action = 'HOLD'
                consensus_confidence = 0.5
                reference_signal = list(signals.values())[0]
            
            # Aggregate reasoning
            active_strategies = [k for k, v in active_signals.items() if v.action == consensus_action]
            urgency = max([s.urgency for s in active_signals.values()], default='low')
            news_catalyst = any(s.news_catalyst for s in active_signals.values())
            
            return TradeSignal(
                symbol=reference_signal.symbol,
                action=consensus_action,
                confidence=min(0.95, consensus_confidence),
                entry_price=reference_signal.entry_price,
                stop_loss=reference_signal.stop_loss,
                take_profit=reference_signal.take_profit,
                position_size=reference_signal.position_size,
                strategy_name=f"advanced_consensus_{'+'.join(active_strategies)}",
                reasoning=f"Consensus from {len(active_strategies)} strategies: {', '.join(active_strategies)}",
                risk_reward_ratio=reference_signal.risk_reward_ratio,
                urgency=urgency,
                news_catalyst=news_catalyst,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Error combining signals: {e}")
            return list(signals.values())[0]
    
    async def close(self):
        """Cleanup resources"""
        if self.news_sentinel:
            await self.news_sentinel.close()