"""
Futures Trader - Main trading logic and execution management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from tradingagents.dataflows.futures_utils import (
    fetch_futures_data, 
    calculate_atr, 
    analyze_market_structure,
    calculate_position_size,
    check_rollover_dates,
    validate_trading_hours
)
from tradingagents.agents.risk_mgmt.futures_risk_manager import FuturesRiskManager

logger = logging.getLogger(__name__)

@dataclass
class TradeSignal:
    """Trade signal data structure."""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: int
    confidence: float
    timestamp: datetime

class FuturesTrader:
    """Main futures trading class."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_manager = FuturesRiskManager(config)
        self.positions = {}  # Only active positions
        self.closed_positions = {}  # Recently closed positions (limited)
        self.trade_history = []  # Complete historical record
        self.daily_pnl = 0.0
        self.account_balance = config.get("initial_balance", 100000)
        self.max_closed_positions = config.get("max_closed_positions", 1000)
        
    def analyze_market(self, symbol: str, timeframe: str = "1d") -> Dict[str, Any]:
        """
        Perform comprehensive market analysis.
        
        Args:
            symbol: Futures symbol to analyze
            timeframe: Data timeframe (1d, 1h, etc.)
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Fetch market data
            data = fetch_futures_data(symbol, period="3mo")
            
            if len(data) < 50:
                return {"error": "Insufficient data for analysis"}
            
            # Calculate technical indicators
            atr = calculate_atr(data)
            current_atr = atr.iloc[-1]
            
            # Market structure analysis
            market_analysis = analyze_market_structure(data)
            
            # Fixed: Safe momentum calculation with validation
            current_close = data['Close'].iloc[-1]
            previous_close = data['Close'].iloc[-20]
            
            if previous_close <= 0 or pd.isna(previous_close):
                logger.warning(f"Invalid previous close price for {symbol}: {previous_close}")
                momentum = 0.0  # Default to no momentum
            else:
                momentum = (current_close / previous_close) - 1
            
            # Additional validation for current price
            if current_close <= 0 or pd.isna(current_close):
                return {"error": f"Invalid current price for {symbol}: {current_close}"}
            
            # Volume analysis
            avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            volume_spike = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Support and resistance levels
            high_20 = data['High'].rolling(window=20).max().iloc[-1]
            low_20 = data['Low'].rolling(window=20).min().iloc[-1]
            
            return {
                "symbol": symbol,
                "current_price": data['Close'].iloc[-1],
                "atr": current_atr,
                "momentum": momentum,
                "volume_spike": volume_spike,
                "support": low_20,
                "resistance": high_20,
                "market_trend": market_analysis.get("market_trend", 1.0),
                "volatility": market_analysis.get("volatility", 0.0),
                "analysis_time": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Market analysis failed for {symbol}: {e}")
            return {"error": str(e)}
    
    def generate_trade_signal(self, symbol: str) -> Optional[TradeSignal]:
        """
        Generate trading signal based on market analysis.
        
        Args:
            symbol: Symbol to generate signal for
            
        Returns:
            TradeSignal object or None if no signal
        """
        # Check trading hours
        if not validate_trading_hours(symbol):
            logger.info(f"Outside trading hours for {symbol}")
            return None
        
        # Get market analysis
        analysis = self.analyze_market(symbol)
        
        if "error" in analysis:
            logger.warning(f"Cannot generate signal for {symbol}: {analysis['error']}")
            return None
        
        current_price = analysis["current_price"]
        atr = analysis["atr"]
        momentum = analysis["momentum"]
        volume_spike = analysis["volume_spike"]
        market_trend = analysis["market_trend"]
        
        # Simple trading logic
        confidence = 0.0
        direction = None
        
        # Fixed: Consistent momentum-based trading logic
        if momentum > 0.02 and market_trend > 1.01:  # Upward momentum
            direction = "long"
            confidence += 0.3
        elif momentum < -0.02 and market_trend < 0.99:  # Downward momentum
            direction = "short"
            confidence += 0.3
        
        # Volume confirmation
        if volume_spike > 1.5:
            confidence += 0.2
        
        # Fixed: Strong trend confirmation (same direction as momentum)
        if market_trend > 1.02 and direction == "long":  # Strong uptrend confirmation
            confidence += 0.4
        elif market_trend < 0.98 and direction == "short":  # Strong downtrend confirmation
            confidence += 0.4
        
        # Check minimum confidence
        if confidence < 0.5 or direction is None:
            return None
        
        # Calculate position size
        contract_specs = self.config.get("contract_specs", {})
        if symbol not in contract_specs:
            logger.error(f"No contract specs for {symbol}")
            return None
        
        spec = contract_specs[symbol]
        contract_value = current_price * spec["contract_size"]
        
        position_size = calculate_position_size(
            account_balance=self.account_balance,
            risk_per_trade=0.02,  # 2% risk
            atr=atr,
            contract_value=contract_value
        )
        
        if position_size == 0:
            return None
        
        # Calculate stop loss and take profit
        stop_loss = self.risk_manager.calculate_stop_loss(
            symbol=symbol,
            entry_price=current_price,
            position_type=direction,
            atr=atr
        )
        
        # Take profit at 2x stop distance
        if direction == "long":
            take_profit = current_price + (current_price - stop_loss) * 2
        else:
            take_profit = current_price - (stop_loss - current_price) * 2
        
        return TradeSignal(
            symbol=symbol,
            direction=direction,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            confidence=confidence,
            timestamp=datetime.now()
        )
    
    def execute_trade(self, signal: TradeSignal) -> Dict[str, Any]:
        """
        Execute a trade based on signal.
        
        Args:
            signal: TradeSignal to execute
            
        Returns:
            Execution result
        """
        # Risk assessment
        risk_assessment = self.risk_manager.assess_position_risk(
            symbol=signal.symbol,
            position_size=signal.position_size,
            current_price=signal.entry_price,
            account_balance=self.account_balance
        )
        
        if not risk_assessment.get("approved", False):
            return {
                "status": "rejected",
                "reason": "Risk assessment failed",
                "risk_flags": risk_assessment.get("risk_flags", [])
            }
        
        # Check rollover requirements
        rollover_info = check_rollover_dates(signal.symbol)
        if rollover_info.get("needs_rollover", False):
            logger.warning(f"Contract {signal.symbol} needs rollover in {rollover_info['days_to_expiry']} days")
        
        # Simulate trade execution
        trade_id = f"{signal.symbol}_{int(signal.timestamp.timestamp())}"
        
        position = {
            "trade_id": trade_id,
            "symbol": signal.symbol,
            "direction": signal.direction,
            "entry_price": signal.entry_price,
            "position_size": signal.position_size,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "entry_time": signal.timestamp,
            "status": "open"
        }
        
        # Fixed: Only store active positions and maintain history
        self.positions[trade_id] = position
        
        # Store in trade history for permanent record
        self.trade_history.append(position.copy())
        
        # Update account balance (simplified)
        margin_required = risk_assessment["margin_required"]
        self.account_balance -= margin_required
        
        logger.info(f"Executed trade: {trade_id} - {signal.direction} {signal.position_size} {signal.symbol} @ {signal.entry_price}")
        
        return {
            "status": "executed",
            "trade_id": trade_id,
            "margin_used": margin_required,
            "remaining_balance": self.account_balance
        }
    
    def _close_position(self, trade_id: str, exit_price: float, exit_time: datetime):
        """Close a position and manage memory cleanup."""
        if trade_id not in self.positions:
            return
        
        position = self.positions[trade_id]
        position["status"] = "closed"
        position["exit_price"] = exit_price
        position["exit_time"] = exit_time
        
        # Move to closed positions with size limit
        self.closed_positions[trade_id] = position
        
        # Remove from active positions
        del self.positions[trade_id]
        
        # Cleanup old closed positions if limit exceeded
        if len(self.closed_positions) > self.max_closed_positions:
            # Remove oldest closed positions
            oldest_trades = sorted(
                self.closed_positions.items(),
                key=lambda x: x[1]["exit_time"]
            )
            
            # Keep only the most recent max_closed_positions
            trades_to_remove = oldest_trades[:-self.max_closed_positions]
            for old_trade_id, _ in trades_to_remove:
                del self.closed_positions[old_trade_id]
                logger.debug(f"Cleaned up old position: {old_trade_id}")
    
    def update_positions(self, market_data: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Update all open positions with current market prices.
        
        Args:
            market_data: Dictionary of symbol -> current_price
            
        Returns:
            List of position updates
        """
        updates = []
        positions_to_close = []
        
        for trade_id, position in self.positions.items():
            symbol = position["symbol"]
            if symbol not in market_data:
                continue
            
            current_price = market_data[symbol]
            entry_price = position["entry_price"]
            direction = position["direction"]
            size = position["position_size"]
            
            # Calculate unrealized P&L
            if direction == "long":
                pnl = (current_price - entry_price) * size
            else:
                pnl = (entry_price - current_price) * size
            
            # Check stop loss and take profit
            stopped_out = False
            if direction == "long":
                if current_price <= position["stop_loss"]:
                    stopped_out = True
                elif current_price >= position["take_profit"]:
                    stopped_out = True
            else:
                if current_price >= position["stop_loss"]:
                    stopped_out = True
                elif current_price <= position["take_profit"]:
                    stopped_out = True
            
            if stopped_out:
                # Mark for closure instead of modifying dict during iteration
                positions_to_close.append((trade_id, current_price, pnl, size, symbol))
                
                updates.append({
                    "trade_id": trade_id,
                    "action": "closed",
                    "exit_price": current_price,
                    "pnl": pnl
                })
            else:
                position["unrealized_pnl"] = pnl
                updates.append({
                    "trade_id": trade_id,
                    "action": "updated",
                    "current_price": current_price,
                    "unrealized_pnl": pnl
                })
        
        # Close positions after iteration
        for trade_id, exit_price, pnl, size, symbol in positions_to_close:
            # Update daily P&L and account balance
            self.daily_pnl += pnl
            
            # Return margin to account
            spec = self.config["contract_specs"][symbol]
            margin_returned = size * spec["margin_req"]
            self.account_balance += margin_returned
            
            # Close the position with proper cleanup
            self._close_position(trade_id, exit_price, datetime.now())
        
        return updates
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary with memory-efficient calculations."""
        open_positions = list(self.positions.values())
        recent_closed = list(self.closed_positions.values())
        
        total_unrealized = sum(p.get("unrealized_pnl", 0) for p in open_positions)
        recent_realized = sum(p.get("realized_pnl", 0) for p in recent_closed)
        
        return {
            "account_balance": self.account_balance,
            "open_positions": len(open_positions),
            "recent_closed_positions": len(recent_closed),
            "total_trade_history": len(self.trade_history),
            "daily_pnl": self.daily_pnl,
            "total_unrealized": total_unrealized,
            "recent_realized": recent_realized,
            "memory_usage": {
                "active_positions": len(self.positions),
                "cached_closed": len(self.closed_positions),
                "total_history": len(self.trade_history)
            }
        }