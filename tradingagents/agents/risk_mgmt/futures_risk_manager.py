"""
Futures Risk Manager for position sizing and risk assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FuturesRiskManager:
    """Risk management for futures trading."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_config = config.get("risk_management", {})
        self.contract_specs = config.get("contract_specs", {})
        
    def assess_position_risk(self, symbol: str, position_size: int, 
                           current_price: float, account_balance: float) -> Dict[str, Any]:
        """
        Assess risk for a proposed position.
        
        Args:
            symbol: Futures symbol
            position_size: Number of contracts
            current_price: Current market price
            account_balance: Available account balance
            
        Returns:
            Risk assessment dictionary
        """
        if symbol not in self.contract_specs:
            return {"error": f"Unknown symbol: {symbol}"}
        
        spec = self.contract_specs[symbol]
        
        # Calculate position value
        contract_size = spec["contract_size"]
        position_value = position_size * current_price * contract_size
        
        # Fixed: Use actual margin requirements from contract specifications
        margin_required = abs(position_size) * spec["margin_req"]
        
        # Position size as percentage of account
        position_pct = position_value / account_balance
        
        # Risk metrics
        max_position_pct = self.risk_config.get("max_position_pct", 0.1)
        max_leverage = self.risk_config.get("max_leverage", 10.0)
        
        # Calculate leverage
        actual_leverage = position_value / account_balance
        
        # Risk flags
        risk_flags = []
        
        if position_pct > max_position_pct:
            risk_flags.append(f"Position size {position_pct:.1%} exceeds maximum {max_position_pct:.1%}")
        
        if actual_leverage > max_leverage:
            risk_flags.append(f"Leverage {actual_leverage:.1f}x exceeds maximum {max_leverage}x")
        
        # Bug #2: Not checking margin availability
        # BUG: Should check if account has sufficient margin
        if margin_required > account_balance:
            risk_flags.append("Insufficient margin")
        
        return {
            "position_value": position_value,
            "margin_required": margin_required,
            "position_pct": position_pct,
            "leverage": actual_leverage,
            "risk_flags": risk_flags,
            "approved": len(risk_flags) == 0
        }
    
    def calculate_stop_loss(self, symbol: str, entry_price: float, 
                          position_type: str, atr: float) -> Optional[float]:
        """
        Calculate stop loss level based on ATR.
        
        Args:
            symbol: Futures symbol
            entry_price: Entry price for position
            position_type: 'long' or 'short'
            atr: Average True Range value
            
        Returns:
            Stop loss price level or None if invalid input
            
        Raises:
            ValueError: If position_type is invalid
        """
        # Input validation
        if not isinstance(position_type, str) or not position_type.strip():
            raise ValueError("Position type must be a non-empty string")
        
        if entry_price <= 0:
            raise ValueError("Entry price must be positive")
        
        if atr <= 0:
            raise ValueError("ATR must be positive")
        
        atr_multiplier = self.risk_config.get("atr_multiplier", 2.0)
        stop_distance = atr * atr_multiplier
        
        position_type_clean = position_type.lower().strip()
        
        if position_type_clean == "long":
            stop_loss = entry_price - stop_distance
        elif position_type_clean == "short":
            stop_loss = entry_price + stop_distance
        else:
            # Fixed: Proper error handling with clear error message
            valid_types = ["long", "short"]
            raise ValueError(f"Invalid position_type '{position_type}'. Must be one of: {valid_types}")
        
        # Additional validation - ensure stop loss makes sense
        if stop_loss <= 0:
            logger.warning(f"Calculated stop loss {stop_loss} is non-positive for {symbol}")
            return None
        
        return stop_loss
    
    def monitor_daily_pnl(self, current_pnl: float, account_balance: float) -> Dict[str, Any]:
        """
        Monitor daily P&L against limits.
        
        Args:
            current_pnl: Current day's P&L
            account_balance: Account balance
            
        Returns:
            P&L monitoring results
        """
        daily_loss_limit = self.risk_config.get("daily_loss_limit", 0.05)
        max_daily_loss = account_balance * daily_loss_limit
        
        pnl_pct = current_pnl / account_balance
        
        warnings = []
        if current_pnl < 0 and abs(current_pnl) > max_daily_loss:
            warnings.append("Daily loss limit exceeded")
        
        # Additional monitoring
        if pnl_pct < -0.02:  # 2% loss warning
            warnings.append("Significant daily loss detected")
        
        return {
            "current_pnl": current_pnl,
            "pnl_pct": pnl_pct,
            "daily_limit": max_daily_loss,
            "warnings": warnings,
            "stop_trading": "Daily loss limit exceeded" in warnings
        }
    
    def validate_margin_requirements(self, positions: List[Dict], 
                                   account_balance: float) -> Dict[str, Any]:
        """
        Validate total margin requirements across all positions.
        
        Args:
            positions: List of position dictionaries
            account_balance: Available account balance
            
        Returns:
            Margin validation results
        """
        total_margin = 0
        margin_buffer = self.risk_config.get("margin_buffer", 0.25)
        
        for position in positions:
            symbol = position.get("symbol")
            size = position.get("size", 0)
            
            if symbol in self.contract_specs:
                margin_per_contract = self.contract_specs[symbol]["margin_req"]
                total_margin += abs(size) * margin_per_contract
        
        # Apply margin buffer
        required_margin = total_margin * (1 + margin_buffer)
        available_margin = account_balance - required_margin
        margin_utilization = required_margin / account_balance
        
        return {
            "total_margin": total_margin,
            "required_margin": required_margin,
            "available_margin": available_margin,
            "margin_utilization": margin_utilization,
            "sufficient_margin": available_margin > 0
        }
    
    def assess_concentration_risk(self, positions: List[Dict]) -> Dict[str, Any]:
        """
        Assess portfolio concentration risk.
        
        Args:
            positions: List of position dictionaries
            
        Returns:
            Concentration risk assessment
        """
        if not positions:
            return {"total_positions": 0, "concentration_risk": "low"}
        
        # Group by symbol and sector
        symbol_exposure = {}
        sector_exposure = {"energy": 0, "metals": 0, "agricultural": 0, 
                          "financial": 0, "currency": 0}
        
        futures_symbols = self.config.get("futures_symbols", {})
        
        for position in positions:
            symbol = position.get("symbol")
            value = position.get("value", 0)
            
            # Symbol concentration
            if symbol in symbol_exposure:
                symbol_exposure[symbol] += abs(value)
            else:
                symbol_exposure[symbol] = abs(value)
            
            # Sector concentration
            for sector, symbols in futures_symbols.items():
                if symbol in symbols:
                    sector_exposure[sector] += abs(value)
                    break
        
        total_exposure = sum(symbol_exposure.values())
        
        # Calculate concentrations
        max_symbol_pct = max(symbol_exposure.values()) / total_exposure if total_exposure > 0 else 0
        max_sector_pct = max(sector_exposure.values()) / total_exposure if total_exposure > 0 else 0
        
        # Risk assessment
        concentration_risk = "low"
        if max_symbol_pct > 0.4 or max_sector_pct > 0.6:
            concentration_risk = "high"
        elif max_symbol_pct > 0.25 or max_sector_pct > 0.4:
            concentration_risk = "medium"
        
        return {
            "total_positions": len(positions),
            "symbol_exposure": symbol_exposure,
            "sector_exposure": sector_exposure,
            "max_symbol_pct": max_symbol_pct,
            "max_sector_pct": max_sector_pct,
            "concentration_risk": concentration_risk
        }