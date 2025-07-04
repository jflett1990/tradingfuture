# Futures-specific technical analysis utilities

import pandas as pd
import numpy as np
from stockstats import wrap
from typing import Annotated, Dict, List, Optional
import os
from .config import get_config
from .futures_utils import FuturesUtils


class FuturesTechnicalUtils:
    
    def __init__(self):
        self.futures_utils = FuturesUtils()
    
    def get_futures_technical_indicators(
        self,
        symbol: Annotated[str, "futures symbol (e.g., CL, GC, ES)"],
        indicators: Annotated[List[str], "list of technical indicators to calculate"],
        curr_date: Annotated[str, "current date for analysis, YYYY-mm-dd"],
        online: Annotated[bool, "whether to fetch data online or use cached data"] = True,
    ) -> Dict:
        """Calculate technical indicators for futures contracts"""
        
        try:
            # Get futures data
            end_date = pd.to_datetime(curr_date)
            start_date = end_date - pd.DateOffset(years=2)  # 2 years for sufficient data
            
            if online:
                # Get config and ensure cache directory exists
                config = get_config()
                os.makedirs(config["data_cache_dir"], exist_ok=True)
                
                cache_file = os.path.join(
                    config["data_cache_dir"],
                    f"{symbol}-futures-data-{start_date.strftime('%Y-%m-%d')}-{end_date.strftime('%Y-%m-%d')}.csv"
                )
                
                if os.path.exists(cache_file):
                    data = pd.read_csv(cache_file)
                    data["Date"] = pd.to_datetime(data["Date"])
                else:
                    data = self.futures_utils.get_futures_data(
                        symbol, 
                        start_date.strftime('%Y-%m-%d'), 
                        end_date.strftime('%Y-%m-%d')
                    )
                    data = data.reset_index()
                    data.to_csv(cache_file, index=False)
            else:
                # Use cached data if available
                config = get_config()
                cache_file = os.path.join(
                    config["data_cache_dir"],
                    f"{symbol}-futures-data-cached.csv"
                )
                
                if os.path.exists(cache_file):
                    data = pd.read_csv(cache_file)
                    data["Date"] = pd.to_datetime(data["Date"])
                else:
                    raise FileNotFoundError(f"Cached data not found for {symbol}")
            
            # Prepare data for stockstats
            df = wrap(data)
            
            # Calculate requested indicators
            results = {}
            for indicator in indicators:
                try:
                    # Trigger calculation
                    df[indicator]
                    
                    # Get the value for the current date
                    curr_date_formatted = pd.to_datetime(curr_date).strftime('%Y-%m-%d')
                    matching_rows = df[df["Date"].dt.strftime('%Y-%m-%d') == curr_date_formatted]
                    
                    if not matching_rows.empty:
                        indicator_value = matching_rows[indicator].iloc[0]
                        results[indicator] = indicator_value
                    else:
                        # Get the latest available value
                        latest_value = df[indicator].dropna().iloc[-1] if not df[indicator].dropna().empty else None
                        results[indicator] = latest_value
                        
                except Exception as e:
                    results[indicator] = f"Error calculating {indicator}: {str(e)}"
            
            return {
                'symbol': symbol,
                'analysis_date': curr_date,
                'indicators': results,
                'data_points': len(data),
                'latest_price': data['Close'].iloc[-1] if not data.empty else None
            }
            
        except Exception as e:
            return {'error': f"Failed to calculate technical indicators for {symbol}: {str(e)}"}
    
    def get_futures_momentum_analysis(
        self,
        symbol: Annotated[str, "futures symbol"],
        curr_date: Annotated[str, "current date, YYYY-mm-dd"],
        online: bool = True,
    ) -> Dict:
        """Perform momentum analysis specific to futures trading"""
        
        momentum_indicators = [
            'rsi',          # RSI for overbought/oversold
            'macd',         # MACD for trend momentum
            'macds',        # MACD signal line
            'macdh',        # MACD histogram
            'cci',          # Commodity Channel Index (great for futures)
            'williams_r',   # Williams %R
            'roc',          # Rate of Change
            'mom',          # Momentum
        ]
        
        try:
            result = self.get_futures_technical_indicators(
                symbol, momentum_indicators, curr_date, online
            )
            
            if 'error' in result:
                return result
            
            indicators = result['indicators']
            
            # Analyze momentum signals
            momentum_signals = []
            
            # RSI analysis
            if 'rsi' in indicators and isinstance(indicators['rsi'], (int, float)):
                rsi_val = indicators['rsi']
                if rsi_val > 70:
                    momentum_signals.append("RSI indicates overbought conditions")
                elif rsi_val < 30:
                    momentum_signals.append("RSI indicates oversold conditions")
                else:
                    momentum_signals.append("RSI in neutral range")
            
            # MACD analysis
            if 'macd' in indicators and 'macds' in indicators:
                macd_val = indicators['macd']
                macds_val = indicators['macds']
                if isinstance(macd_val, (int, float)) and isinstance(macds_val, (int, float)):
                    if macd_val > macds_val:
                        momentum_signals.append("MACD shows bullish momentum")
                    else:
                        momentum_signals.append("MACD shows bearish momentum")
            
            # CCI analysis (particularly relevant for commodities)
            if 'cci' in indicators and isinstance(indicators['cci'], (int, float)):
                cci_val = indicators['cci']
                if cci_val > 100:
                    momentum_signals.append("CCI indicates strong bullish momentum")
                elif cci_val < -100:
                    momentum_signals.append("CCI indicates strong bearish momentum")
                else:
                    momentum_signals.append("CCI in normal range")
            
            result['momentum_analysis'] = {
                'signals': momentum_signals,
                'overall_momentum': self._assess_overall_momentum(indicators),
                'key_levels': self._identify_key_levels(indicators)
            }
            
            return result
            
        except Exception as e:
            return {'error': f"Failed to perform momentum analysis for {symbol}: {str(e)}"}
    
    def get_futures_volatility_analysis(
        self,
        symbol: Annotated[str, "futures symbol"],
        curr_date: Annotated[str, "current date, YYYY-mm-dd"],
        online: bool = True,
    ) -> Dict:
        """Analyze volatility patterns specific to futures"""
        
        volatility_indicators = [
            'atr',          # Average True Range
            'boll',         # Bollinger Bands middle
            'boll_ub',      # Bollinger Upper Band
            'boll_lb',      # Bollinger Lower Band
            'keltner',      # Keltner Channel
            'dpo',          # Detrended Price Oscillator
        ]
        
        try:
            result = self.get_futures_technical_indicators(
                symbol, volatility_indicators, curr_date, online
            )
            
            if 'error' in result:
                return result
            
            indicators = result['indicators']
            
            # Analyze volatility signals
            volatility_signals = []
            
            # ATR analysis
            if 'atr' in indicators and isinstance(indicators['atr'], (int, float)):
                atr_val = indicators['atr']
                volatility_signals.append(f"ATR indicates volatility level of {atr_val:.2f}")
            
            # Bollinger Bands analysis
            if all(k in indicators for k in ['boll', 'boll_ub', 'boll_lb']):
                current_price = result.get('latest_price', 0)
                boll_upper = indicators['boll_ub']
                boll_lower = indicators['boll_lb']
                
                if all(isinstance(x, (int, float)) for x in [current_price, boll_upper, boll_lower]):
                    if current_price > boll_upper:
                        volatility_signals.append("Price above Bollinger upper band - high volatility")
                    elif current_price < boll_lower:
                        volatility_signals.append("Price below Bollinger lower band - high volatility")
                    else:
                        volatility_signals.append("Price within Bollinger bands - normal volatility")
            
            result['volatility_analysis'] = {
                'signals': volatility_signals,
                'volatility_regime': self._assess_volatility_regime(indicators),
                'risk_metrics': self._calculate_risk_metrics(indicators)
            }
            
            return result
            
        except Exception as e:
            return {'error': f"Failed to perform volatility analysis for {symbol}: {str(e)}"}
    
    def _assess_overall_momentum(self, indicators: Dict) -> str:
        """Assess overall momentum direction"""
        bullish_signals = 0
        bearish_signals = 0
        
        # RSI assessment
        if 'rsi' in indicators and isinstance(indicators['rsi'], (int, float)):
            rsi_val = indicators['rsi']
            if rsi_val > 50:
                bullish_signals += 1
            elif rsi_val < 50:
                bearish_signals += 1
        
        # MACD assessment
        if 'macd' in indicators and 'macds' in indicators:
            macd_val = indicators['macd']
            macds_val = indicators['macds']
            if isinstance(macd_val, (int, float)) and isinstance(macds_val, (int, float)):
                if macd_val > macds_val:
                    bullish_signals += 1
                else:
                    bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            return "Bullish"
        elif bearish_signals > bullish_signals:
            return "Bearish"
        else:
            return "Neutral"
    
    def _identify_key_levels(self, indicators: Dict) -> Dict:
        """Identify key support and resistance levels"""
        levels = {}
        
        # Bollinger Bands levels
        if 'boll_ub' in indicators and 'boll_lb' in indicators:
            levels['resistance'] = indicators['boll_ub']
            levels['support'] = indicators['boll_lb']
        
        return levels
    
    def _assess_volatility_regime(self, indicators: Dict) -> str:
        """Assess current volatility regime"""
        if 'atr' in indicators and isinstance(indicators['atr'], (int, float)):
            atr_val = indicators['atr']
            # This is a simplified assessment - in practice, you'd compare to historical ATR
            if atr_val > 2.0:  # Threshold would depend on the specific futures contract
                return "High Volatility"
            elif atr_val < 0.5:
                return "Low Volatility"
            else:
                return "Normal Volatility"
        
        return "Unknown"
    
    def _calculate_risk_metrics(self, indicators: Dict) -> Dict:
        """Calculate risk metrics for futures trading"""
        metrics = {}
        
        if 'atr' in indicators and isinstance(indicators['atr'], (int, float)):
            atr_val = indicators['atr']
            metrics['suggested_stop_loss'] = atr_val * 2  # 2x ATR stop loss
            metrics['position_sizing_factor'] = 1 / atr_val if atr_val > 0 else 1
        
        return metrics