# Futures market data utilities

import yfinance as yf
import pandas as pd
from typing import Annotated, Callable, Any, Dict
from functools import wraps

from .utils import decorate_all_methods


def init_futures_ticker(func: Callable) -> Callable:
    """Decorator to initialize yf.Ticker for futures and pass it to the function."""
    
    @wraps(func)
    def wrapper(symbol: Annotated[str, "futures symbol"], *args, **kwargs) -> Any:
        # Handle futures symbol format (e.g., CL=F for crude oil)
        if not symbol.endswith('=F'):
            symbol = symbol + '=F'
        ticker = yf.Ticker(symbol)
        return func(ticker, *args, **kwargs)
    
    return wrapper


@decorate_all_methods(init_futures_ticker)
class FuturesUtils:
    
    def get_futures_data(
        symbol: Annotated[str, "futures symbol (e.g., CL, GC, ES)"],
        start_date: Annotated[str, "start date for retrieving futures data, YYYY-mm-dd"],
        end_date: Annotated[str, "end date for retrieving futures data, YYYY-mm-dd"],
    ) -> pd.DataFrame:
        """Retrieve futures price data for designated symbol"""
        ticker = symbol
        # Add one day to the end_date so that the data range is inclusive
        end_date = pd.to_datetime(end_date) + pd.DateOffset(days=1)
        end_date = end_date.strftime("%Y-%m-%d")
        
        try:
            futures_data = ticker.history(start=start_date, end=end_date)
            if futures_data.empty:
                raise ValueError(f"No data found for futures symbol {symbol}")
            return futures_data
        except Exception as e:
            raise Exception(f"Failed to fetch futures data for {symbol}: {str(e)}")
    
    def get_futures_info(
        symbol: Annotated[str, "futures symbol"],
    ) -> Dict:
        """Fetches and returns futures contract information."""
        ticker = symbol
        try:
            info = ticker.info
            # Filter futures-specific information
            futures_info = {
                'symbol': info.get('symbol', symbol),
                'shortName': info.get('shortName', 'N/A'),
                'fullName': info.get('longName', 'N/A'),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'N/A'),
                'contractSize': info.get('contractSize', 'N/A'),
                'tickSize': info.get('tickSize', 'N/A'),
                'lastTradeDate': info.get('lastTradeDate', 'N/A'),
                'expiryDate': info.get('expiryDate', 'N/A'),
                'firstTradeDate': info.get('firstTradeDate', 'N/A'),
                'underlyingSymbol': info.get('underlyingSymbol', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
            }
            return futures_info
        except Exception as e:
            return {'error': f"Failed to fetch info for {symbol}: {str(e)}"}
    
    def get_contango_backwardation(
        symbol: Annotated[str, "base futures symbol (e.g., CL, GC)"],
        start_date: Annotated[str, "start date, YYYY-mm-dd"],
        end_date: Annotated[str, "end date, YYYY-mm-dd"],
    ) -> Dict:
        """Analyze contango/backwardation by comparing near and far month contracts"""
        try:
            # Get current front month and next month contracts
            # This is a simplified approach - in practice, you'd need to handle rollover dates
            front_month = self.get_futures_data(symbol, start_date, end_date)
            
            if front_month.empty:
                return {'error': f"No data available for {symbol}"}
            
            # Calculate basic contango/backwardation metrics
            latest_price = front_month['Close'].iloc[-1]
            
            # Simple contango analysis based on price trends
            price_trend = front_month['Close'].pct_change().tail(20).mean()
            
            result = {
                'symbol': symbol,
                'latest_price': latest_price,
                'price_trend_20d': price_trend,
                'market_structure': 'contango' if price_trend > 0 else 'backwardation',
                'analysis_date': end_date,
                'data_points': len(front_month)
            }
            
            return result
            
        except Exception as e:
            return {'error': f"Failed to analyze contango/backwardation for {symbol}: {str(e)}"}
    
    def get_volume_open_interest(
        symbol: Annotated[str, "futures symbol"],
        start_date: Annotated[str, "start date, YYYY-mm-dd"],
        end_date: Annotated[str, "end date, YYYY-mm-dd"],
    ) -> Dict:
        """Get volume and open interest data for futures contract"""
        try:
            data = self.get_futures_data(symbol, start_date, end_date)
            
            if data.empty:
                return {'error': f"No data available for {symbol}"}
            
            # Calculate volume and open interest metrics
            avg_volume = data['Volume'].mean()
            latest_volume = data['Volume'].iloc[-1]
            volume_trend = data['Volume'].pct_change().tail(10).mean()
            
            result = {
                'symbol': symbol,
                'average_volume': avg_volume,
                'latest_volume': latest_volume,
                'volume_trend_10d': volume_trend,
                'total_volume': data['Volume'].sum(),
                'high_volume_days': len(data[data['Volume'] > avg_volume * 1.5]),
                'analysis_period': f"{start_date} to {end_date}"
            }
            
            return result
            
        except Exception as e:
            return {'error': f"Failed to analyze volume/OI for {symbol}: {str(e)}"}
    
    def get_futures_curve_analysis(
        base_symbol: Annotated[str, "base futures symbol (e.g., CL, GC)"],
        date: Annotated[str, "analysis date, YYYY-mm-dd"],
    ) -> Dict:
        """Analyze futures curve structure"""
        try:
            # This is a simplified implementation
            # In practice, you'd need to fetch multiple contract months
            data = self.get_futures_data(base_symbol, date, date)
            
            if data.empty:
                return {'error': f"No data available for {base_symbol} on {date}"}
            
            result = {
                'symbol': base_symbol,
                'analysis_date': date,
                'front_month_price': data['Close'].iloc[-1] if not data.empty else None,
                'curve_analysis': 'Limited - single contract analysis',
                'recommendation': 'Use multiple contract months for full curve analysis'
            }
            
            return result
            
        except Exception as e:
            return {'error': f"Failed to analyze futures curve for {base_symbol}: {str(e)}"}
    
    def get_margin_requirements(
        symbol: Annotated[str, "futures symbol"],
    ) -> Dict:
        """Get estimated margin requirements (placeholder - would need broker API)"""
        # This would typically require integration with broker APIs
        # Providing estimated values for common futures
        
        margin_estimates = {
            'CL': {'initial': 4500, 'maintenance': 4000, 'currency': 'USD'},  # Crude Oil
            'GC': {'initial': 8000, 'maintenance': 7000, 'currency': 'USD'},  # Gold
            'ES': {'initial': 13000, 'maintenance': 11000, 'currency': 'USD'},  # E-mini S&P
            'NQ': {'initial': 17000, 'maintenance': 15000, 'currency': 'USD'},  # E-mini NASDAQ
            'YM': {'initial': 6000, 'maintenance': 5000, 'currency': 'USD'},   # E-mini Dow
            'ZC': {'initial': 1500, 'maintenance': 1200, 'currency': 'USD'},  # Corn
            'ZW': {'initial': 1800, 'maintenance': 1500, 'currency': 'USD'},  # Wheat
        }
        
        base_symbol = symbol.replace('=F', '').upper()
        
        if base_symbol in margin_estimates:
            return {
                'symbol': symbol,
                'estimated_margins': margin_estimates[base_symbol],
                'note': 'Estimated values - check with broker for actual requirements',
                'last_updated': 'Static estimates'
            }
        else:
            return {
                'symbol': symbol,
                'error': 'Margin requirements not available for this symbol',
                'note': 'Contact broker for specific margin requirements'
            }
    
    def get_expiry_analysis(
        symbol: Annotated[str, "futures symbol"],
        current_date: Annotated[str, "current date, YYYY-mm-dd"],
    ) -> Dict:
        """Analyze contract expiry and rollover timing"""
        try:
            info = self.get_futures_info(symbol)
            
            if 'error' in info:
                return info
            
            expiry_date = info.get('expiryDate', 'N/A')
            
            if expiry_date != 'N/A':
                try:
                    if isinstance(expiry_date, (int, float)):
                        expiry_dt = pd.to_datetime(expiry_date, unit='s')
                    else:
                        expiry_dt = pd.to_datetime(expiry_date)
                    
                    current_dt = pd.to_datetime(current_date)
                    days_to_expiry = (expiry_dt - current_dt).days
                    
                    result = {
                        'symbol': symbol,
                        'expiry_date': expiry_dt.strftime('%Y-%m-%d'),
                        'days_to_expiry': days_to_expiry,
                        'rollover_warning': days_to_expiry <= 5,
                        'analysis_date': current_date
                    }
                    
                    if days_to_expiry <= 5:
                        result['recommendation'] = 'Consider rolling to next contract'
                    elif days_to_expiry <= 20:
                        result['recommendation'] = 'Monitor for rollover timing'
                    else:
                        result['recommendation'] = 'Contract has sufficient time to expiry'
                    
                    return result
                    
                except Exception as e:
                    return {'error': f"Failed to parse expiry date: {str(e)}"}
            else:
                return {'error': 'Expiry date not available'}
                
        except Exception as e:
            return {'error': f"Failed to analyze expiry for {symbol}: {str(e)}"}
    
    def get_futures_fundamentals(
        symbol: Annotated[str, "futures symbol"],
        date: Annotated[str, "analysis date, YYYY-mm-dd"],
    ) -> Dict:
        """Get fundamental data relevant to futures (supply/demand, storage, etc.)"""
        # This would integrate with commodity-specific data sources
        # For now, providing a framework
        
        base_symbol = symbol.replace('=F', '').upper()
        
        fundamentals_framework = {
            'CL': {  # Crude Oil
                'key_factors': ['EIA inventory', 'OPEC production', 'Refinery utilization'],
                'seasonal_patterns': 'Driving season (summer), Heating season (winter)',
                'supply_demand': 'Global production vs consumption',
                'storage': 'Cushing, OK inventory levels'
            },
            'GC': {  # Gold
                'key_factors': ['USD strength', 'Interest rates', 'Inflation', 'Geopolitical events'],
                'seasonal_patterns': 'Wedding seasons, Central bank buying',
                'supply_demand': 'Mine production vs jewelry/investment demand',
                'storage': 'ETF holdings, Central bank reserves'
            },
            'ZC': {  # Corn
                'key_factors': ['Weather', 'Planting/Harvest', 'Export demand', 'Ethanol production'],
                'seasonal_patterns': 'Planting (spring), Growing season, Harvest (fall)',
                'supply_demand': 'US production vs feed/export/ethanol demand',
                'storage': 'USDA inventory reports'
            }
        }
        
        if base_symbol in fundamentals_framework:
            return {
                'symbol': symbol,
                'analysis_date': date,
                'fundamentals': fundamentals_framework[base_symbol],
                'note': 'Framework provided - integrate with actual data sources'
            }
        else:
            return {
                'symbol': symbol,
                'error': 'Fundamental analysis framework not available for this symbol',
                'recommendation': 'Research commodity-specific fundamental factors'
            }