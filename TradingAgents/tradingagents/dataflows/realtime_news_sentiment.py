# Real-time News Sentiment Analysis for Futures Markets

import asyncio
import aiohttp
import feedparser
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import json
from textblob import TextBlob
import yfinance as yf
from newsapi import NewsApiClient
import tweepy
import warnings
warnings.filterwarnings('ignore')


@dataclass
class NewsItem:
    title: str
    content: str
    source: str
    timestamp: datetime
    sentiment_score: float
    sentiment_label: str
    urgency_score: float
    futures_relevance: float
    symbols_mentioned: List[str]
    category: str


class RealTimeNewsSentiment:
    def __init__(self, config: Dict):
        self.config = config
        self.news_sources = self._initialize_news_sources()
        self.futures_keywords = self._initialize_futures_keywords()
        self.sentiment_cache = {}
        self.cache_duration = 300  # 5 minutes
        self.news_buffer = []
        self.max_buffer_size = 1000
        self.running = False
        self.update_interval = 10  # seconds
        
        # Initialize APIs
        self.news_api = self._initialize_news_api()
        self.twitter_api = self._initialize_twitter_api()
        
    def _initialize_news_sources(self) -> Dict:
        """Initialize news sources with futures-specific feeds"""
        return {
            'energy': {
                'rss_feeds': [
                    'https://www.energyintel.com/rss/news',
                    'https://www.oilprice.com/rss/main',
                    'https://www.platts.com/rss/xml/oil',
                    'https://www.reuters.com/rssFeed/businessNews/energy',
                ],
                'keywords': ['crude oil', 'WTI', 'Brent', 'natural gas', 'gasoline', 'heating oil']
            },
            'metals': {
                'rss_feeds': [
                    'https://www.kitco.com/rss/KitcoNews.xml',
                    'https://www.mining.com/rss/',
                    'https://www.reuters.com/rssFeed/businessNews/metals',
                ],
                'keywords': ['gold', 'silver', 'platinum', 'palladium', 'copper', 'precious metals']
            },
            'agriculture': {
                'rss_feeds': [
                    'https://www.agweb.com/rss/news',
                    'https://www.agriculture.com/rss/news',
                    'https://www.reuters.com/rssFeed/businessNews/agriculture',
                ],
                'keywords': ['corn', 'soybeans', 'wheat', 'coffee', 'sugar', 'cotton', 'weather', 'harvest']
            },
            'financial': {
                'rss_feeds': [
                    'https://www.reuters.com/rssFeed/businessNews/markets',
                    'https://feeds.bloomberg.com/markets/news.rss',
                    'https://www.marketwatch.com/rss/realtimeheadlines',
                ],
                'keywords': ['S&P 500', 'NASDAQ', 'Dow Jones', 'futures', 'fed', 'interest rates', 'inflation']
            },
            'general': {
                'rss_feeds': [
                    'https://www.cnbc.com/id/100003114/device/rss/rss.html',
                    'https://www.wsj.com/xml/rss/3_7085.xml',
                    'https://www.ft.com/rss/home',
                ],
                'keywords': ['breaking news', 'market', 'economy', 'central bank', 'trade war']
            }
        }
    
    def _initialize_futures_keywords(self) -> Dict:
        """Initialize futures-specific keywords and their symbols"""
        return {
            'CL': ['crude oil', 'WTI', 'oil price', 'petroleum', 'API inventory', 'EIA', 'OPEC'],
            'NG': ['natural gas', 'LNG', 'gas price', 'storage', 'pipeline', 'weather'],
            'GC': ['gold', 'gold price', 'bullion', 'safe haven', 'USD', 'inflation'],
            'SI': ['silver', 'silver price', 'precious metals', 'industrial demand'],
            'ZC': ['corn', 'corn price', 'ethanol', 'weather', 'crop report', 'USDA'],
            'ZS': ['soybeans', 'soybean', 'China trade', 'crop', 'export'],
            'ZW': ['wheat', 'wheat price', 'grain', 'weather', 'harvest'],
            'ES': ['S&P 500', 'SPX', 'equity futures', 'stock market', 'fed'],
            'NQ': ['NASDAQ', 'tech stocks', 'technology', 'growth stocks'],
            'YM': ['Dow Jones', 'DJIA', 'industrial stocks'],
            '6E': ['EUR/USD', 'euro', 'ECB', 'eurozone', 'European Central Bank'],
            '6J': ['USD/JPY', 'yen', 'Bank of Japan', 'BOJ', 'Japan'],
        }
    
    def _initialize_news_api(self) -> Optional[NewsApiClient]:
        """Initialize News API client"""
        api_key = self.config.get('news_api_key')
        if api_key:
            return NewsApiClient(api_key=api_key)
        return None
    
    def _initialize_twitter_api(self) -> Optional[tweepy.API]:
        """Initialize Twitter API client"""
        twitter_config = self.config.get('twitter_api', {})
        if all(k in twitter_config for k in ['consumer_key', 'consumer_secret', 'access_token', 'access_token_secret']):
            auth = tweepy.OAuthHandler(twitter_config['consumer_key'], twitter_config['consumer_secret'])
            auth.set_access_token(twitter_config['access_token'], twitter_config['access_token_secret'])
            return tweepy.API(auth, wait_on_rate_limit=True)
        return None
    
    async def fetch_rss_feed(self, url: str) -> List[NewsItem]:
        """Fetch and parse RSS feed asynchronously"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        return self._parse_rss_content(content, url)
        except Exception as e:
            print(f"Error fetching RSS feed {url}: {e}")
        return []
    
    def _parse_rss_content(self, content: str, source_url: str) -> List[NewsItem]:
        """Parse RSS content and extract news items"""
        try:
            feed = feedparser.parse(content)
            news_items = []
            
            for entry in feed.entries[:20]:  # Limit to recent 20 items
                # Extract relevant information
                title = entry.get('title', '')
                summary = entry.get('summary', entry.get('description', ''))
                pub_date = entry.get('published_parsed')
                
                if pub_date:
                    timestamp = datetime(*pub_date[:6])
                else:
                    timestamp = datetime.now()
                
                # Only process recent news (last 24 hours)
                if timestamp > datetime.now() - timedelta(hours=24):
                    news_item = self._create_news_item(title, summary, source_url, timestamp)
                    if news_item:
                        news_items.append(news_item)
            
            return news_items
        except Exception as e:
            print(f"Error parsing RSS content: {e}")
            return []
    
    def _create_news_item(self, title: str, content: str, source: str, timestamp: datetime) -> Optional[NewsItem]:
        """Create a NewsItem with sentiment analysis"""
        try:
            # Combine title and content for analysis
            full_text = f"{title} {content}"
            
            # Sentiment analysis
            sentiment_score, sentiment_label = self._analyze_sentiment(full_text)
            
            # Calculate urgency score
            urgency_score = self._calculate_urgency_score(title, content)
            
            # Identify mentioned futures symbols
            symbols_mentioned = self._identify_futures_symbols(full_text)
            
            # Calculate futures relevance
            futures_relevance = self._calculate_futures_relevance(full_text, symbols_mentioned)
            
            # Categorize news
            category = self._categorize_news(full_text)
            
            # Only return if relevant to futures trading
            if futures_relevance > 0.3 or symbols_mentioned:
                return NewsItem(
                    title=title,
                    content=content,
                    source=source,
                    timestamp=timestamp,
                    sentiment_score=sentiment_score,
                    sentiment_label=sentiment_label,
                    urgency_score=urgency_score,
                    futures_relevance=futures_relevance,
                    symbols_mentioned=symbols_mentioned,
                    category=category
                )
        except Exception as e:
            print(f"Error creating news item: {e}")
        
        return None
    
    def _analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """Analyze sentiment using TextBlob with futures-specific adjustments"""
        try:
            # Basic sentiment analysis
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # Adjust for futures-specific terms
            futures_sentiment_adjustments = {
                'bullish': 0.3, 'bearish': -0.3, 'rally': 0.4, 'crash': -0.5,
                'surge': 0.4, 'plunge': -0.4, 'soar': 0.5, 'tumble': -0.4,
                'breakout': 0.3, 'breakdown': -0.3, 'squeeze': 0.2,
                'inventory build': -0.2, 'inventory draw': 0.2,
                'hawkish': -0.3, 'dovish': 0.3, 'tightening': -0.3,
                'easing': 0.3, 'stimulus': 0.4, 'rate hike': -0.3,
                'weather concern': -0.2, 'supply disruption': 0.3,
                'demand destruction': -0.4, 'strong demand': 0.3
            }
            
            text_lower = text.lower()
            for term, adjustment in futures_sentiment_adjustments.items():
                if term in text_lower:
                    polarity += adjustment
            
            # Normalize polarity
            polarity = max(-1, min(1, polarity))
            
            # Determine sentiment label
            if polarity > 0.1:
                sentiment_label = 'bullish'
            elif polarity < -0.1:
                sentiment_label = 'bearish'
            else:
                sentiment_label = 'neutral'
            
            return polarity, sentiment_label
        except:
            return 0.0, 'neutral'
    
    def _calculate_urgency_score(self, title: str, content: str) -> float:
        """Calculate urgency score based on keywords and context"""
        urgency_keywords = {
            'breaking': 0.9, 'urgent': 0.8, 'alert': 0.8, 'flash': 0.9,
            'immediate': 0.7, 'sudden': 0.6, 'unexpected': 0.6,
            'surprise': 0.5, 'shock': 0.7, 'emergency': 0.8,
            'halt': 0.8, 'suspend': 0.7, 'limit up': 0.9, 'limit down': 0.9,
            'circuit breaker': 0.8, 'margin call': 0.7
        }
        
        text_lower = f"{title} {content}".lower()
        max_urgency = 0.0
        
        for keyword, score in urgency_keywords.items():
            if keyword in text_lower:
                max_urgency = max(max_urgency, score)
        
        # Boost urgency for numbers/percentages
        if re.search(r'\d+%', text_lower):
            max_urgency = max(max_urgency, 0.4)
        
        return max_urgency
    
    def _identify_futures_symbols(self, text: str) -> List[str]:
        """Identify futures symbols mentioned in the text"""
        symbols = []
        text_lower = text.lower()
        
        for symbol, keywords in self.futures_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    if symbol not in symbols:
                        symbols.append(symbol)
                    break
        
        return symbols
    
    def _calculate_futures_relevance(self, text: str, symbols_mentioned: List[str]) -> float:
        """Calculate how relevant the news is to futures trading"""
        relevance_score = 0.0
        
        # Base relevance from mentioned symbols
        relevance_score += len(symbols_mentioned) * 0.3
        
        # Futures-specific terms
        futures_terms = [
            'futures', 'commodity', 'contract', 'expiry', 'rollover',
            'margin', 'leverage', 'contango', 'backwardation',
            'inventory', 'supply', 'demand', 'production'
        ]
        
        text_lower = text.lower()
        for term in futures_terms:
            if term in text_lower:
                relevance_score += 0.1
        
        # Market-moving terms
        market_terms = [
            'fed', 'central bank', 'interest rate', 'inflation',
            'gdp', 'unemployment', 'trade war', 'tariff',
            'weather', 'drought', 'flood', 'hurricane'
        ]
        
        for term in market_terms:
            if term in text_lower:
                relevance_score += 0.15
        
        return min(1.0, relevance_score)
    
    def _categorize_news(self, text: str) -> str:
        """Categorize news into futures sectors"""
        text_lower = text.lower()
        
        for category, data in self.news_sources.items():
            if category == 'general':
                continue
            for keyword in data['keywords']:
                if keyword.lower() in text_lower:
                    return category
        
        return 'general'
    
    async def fetch_news_api_data(self, symbol: str) -> List[NewsItem]:
        """Fetch news from News API for specific symbol"""
        if not self.news_api:
            return []
        
        try:
            # Get keywords for the symbol
            keywords = self.futures_keywords.get(symbol, [])
            if not keywords:
                return []
            
            # Fetch news for each keyword
            news_items = []
            for keyword in keywords[:3]:  # Limit to top 3 keywords
                articles = self.news_api.get_everything(
                    q=keyword,
                    language='en',
                    sort_by='publishedAt',
                    page_size=20,
                    from_param=(datetime.now() - timedelta(hours=6)).isoformat()
                )
                
                for article in articles.get('articles', []):
                    news_item = self._create_news_item(
                        article['title'],
                        article.get('description', ''),
                        article['source']['name'],
                        datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                    )
                    if news_item:
                        news_items.append(news_item)
            
            return news_items
        except Exception as e:
            print(f"Error fetching News API data: {e}")
            return []
    
    def fetch_twitter_sentiment(self, symbol: str) -> Dict:
        """Fetch Twitter sentiment for futures symbol"""
        if not self.twitter_api:
            return {'sentiment': 'neutral', 'volume': 0, 'score': 0.0}
        
        try:
            keywords = self.futures_keywords.get(symbol, [])
            if not keywords:
                return {'sentiment': 'neutral', 'volume': 0, 'score': 0.0}
            
            # Search for recent tweets
            tweets = []
            for keyword in keywords[:2]:  # Limit to top 2 keywords
                try:
                    tweet_results = tweepy.Cursor(
                        self.twitter_api.search_tweets,
                        q=keyword + " -RT",
                        lang="en",
                        result_type="recent",
                        tweet_mode="extended"
                    ).items(50)
                    
                    tweets.extend(tweet_results)
                except:
                    continue
            
            if not tweets:
                return {'sentiment': 'neutral', 'volume': 0, 'score': 0.0}
            
            # Analyze sentiment
            sentiments = []
            for tweet in tweets:
                sentiment_score, _ = self._analyze_sentiment(tweet.full_text)
                sentiments.append(sentiment_score)
            
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            if avg_sentiment > 0.1:
                sentiment_label = 'bullish'
            elif avg_sentiment < -0.1:
                sentiment_label = 'bearish'
            else:
                sentiment_label = 'neutral'
            
            return {
                'sentiment': sentiment_label,
                'volume': len(tweets),
                'score': avg_sentiment
            }
        except Exception as e:
            print(f"Error fetching Twitter sentiment: {e}")
            return {'sentiment': 'neutral', 'volume': 0, 'score': 0.0}
    
    async def get_real_time_sentiment(self, symbol: str) -> Dict:
        """Get real-time sentiment analysis for a futures symbol"""
        # Check cache first
        cache_key = f"{symbol}_{int(time.time() / self.cache_duration)}"
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        # Fetch from multiple sources
        tasks = []
        
        # RSS feeds
        for category, data in self.news_sources.items():
            for feed_url in data['rss_feeds']:
                tasks.append(self.fetch_rss_feed(feed_url))
        
        # News API
        if self.news_api:
            tasks.append(self.fetch_news_api_data(symbol))
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine all news items
        all_news = []
        for result in results:
            if isinstance(result, list):
                all_news.extend(result)
        
        # Filter for symbol-specific news
        relevant_news = [
            item for item in all_news 
            if symbol in item.symbols_mentioned or item.futures_relevance > 0.5
        ]
        
        # Get Twitter sentiment
        twitter_sentiment = self.fetch_twitter_sentiment(symbol)
        
        # Calculate overall sentiment
        sentiment_analysis = self._calculate_overall_sentiment(relevant_news, twitter_sentiment)
        
        # Cache the result
        self.sentiment_cache[cache_key] = sentiment_analysis
        
        return sentiment_analysis
    
    def _calculate_overall_sentiment(self, news_items: List[NewsItem], twitter_data: Dict) -> Dict:
        """Calculate overall sentiment from multiple sources"""
        if not news_items and twitter_data['volume'] == 0:
            return {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'news_count': 0,
                'twitter_volume': 0,
                'urgency_level': 'low',
                'key_drivers': [],
                'timestamp': datetime.now()
            }
        
        # Weight news sentiment by urgency and recency
        news_sentiments = []
        key_drivers = []
        
        for item in news_items:
            # Weight by urgency and recency
            hours_old = (datetime.now() - item.timestamp).total_seconds() / 3600
            recency_weight = max(0.1, 1 - (hours_old / 24))  # Decay over 24 hours
            urgency_weight = 1 + item.urgency_score
            
            weighted_sentiment = item.sentiment_score * recency_weight * urgency_weight
            news_sentiments.append(weighted_sentiment)
            
            # Track key drivers
            if item.urgency_score > 0.5 or abs(item.sentiment_score) > 0.3:
                key_drivers.append({
                    'title': item.title,
                    'sentiment': item.sentiment_label,
                    'urgency': item.urgency_score,
                    'timestamp': item.timestamp
                })
        
        # Combine news and Twitter sentiment
        news_weight = 0.7
        twitter_weight = 0.3
        
        if news_sentiments:
            avg_news_sentiment = sum(news_sentiments) / len(news_sentiments)
        else:
            avg_news_sentiment = 0.0
            news_weight = 0.0
            twitter_weight = 1.0
        
        overall_score = (
            avg_news_sentiment * news_weight + 
            twitter_data['score'] * twitter_weight
        )
        
        # Determine overall sentiment
        if overall_score > 0.15:
            overall_sentiment = 'bullish'
        elif overall_score < -0.15:
            overall_sentiment = 'bearish'
        else:
            overall_sentiment = 'neutral'
        
        # Calculate confidence based on data volume and consistency
        confidence = min(1.0, 
            (len(news_items) + twitter_data['volume'] / 10) / 20 * 
            (1 - abs(overall_score - avg_news_sentiment))
        )
        
        # Determine urgency level
        max_urgency = max([item.urgency_score for item in news_items] + [0])
        if max_urgency > 0.7:
            urgency_level = 'high'
        elif max_urgency > 0.4:
            urgency_level = 'medium'
        else:
            urgency_level = 'low'
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_score': overall_score,
            'confidence': confidence,
            'news_count': len(news_items),
            'twitter_volume': twitter_data['volume'],
            'urgency_level': urgency_level,
            'key_drivers': key_drivers[:5],  # Top 5 key drivers
            'timestamp': datetime.now()
        }
    
    def start_real_time_monitoring(self, symbols: List[str]):
        """Start real-time monitoring for multiple symbols"""
        self.running = True
        
        def monitor_loop():
            while self.running:
                try:
                    for symbol in symbols:
                        asyncio.run(self.get_real_time_sentiment(symbol))
                    time.sleep(self.update_interval)
                except Exception as e:
                    print(f"Error in monitoring loop: {e}")
                    time.sleep(self.update_interval)
        
        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def stop_real_time_monitoring(self):
        """Stop real-time monitoring"""
        self.running = False
    
    def get_sentiment_alerts(self, threshold: float = 0.3) -> List[Dict]:
        """Get sentiment alerts for significant changes"""
        alerts = []
        
        for symbol in self.futures_keywords.keys():
            try:
                sentiment_data = asyncio.run(self.get_real_time_sentiment(symbol))
                
                if (abs(sentiment_data['sentiment_score']) > threshold or 
                    sentiment_data['urgency_level'] == 'high'):
                    alerts.append({
                        'symbol': symbol,
                        'sentiment': sentiment_data['overall_sentiment'],
                        'score': sentiment_data['sentiment_score'],
                        'urgency': sentiment_data['urgency_level'],
                        'confidence': sentiment_data['confidence'],
                        'key_drivers': sentiment_data['key_drivers']
                    })
            except Exception as e:
                print(f"Error getting sentiment alert for {symbol}: {e}")
        
        return alerts