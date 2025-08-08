# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # API Configuration - Read from environment variables
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = "gpt-3.5-turbo"
    MAX_TOKENS = 1000
    TEMPERATURE = 0.7
    
    # Portfolio Optimization Configuration
    RISK_FREE_RATE = 0.02
    DEFAULT_DATA_PERIOD = "2y"
    MAX_SINGLE_ASSET_WEIGHT = 0.4
    MIN_ASSET_WEIGHT = 0.01
    REBALANCE_THRESHOLD = 0.05
    
    # Multi-Objective Optimization Configuration
    PARETO_POPULATION_SIZE = 100
    PARETO_GENERATIONS = 50
    PARETO_SOLUTIONS_TO_SHOW = 8  # Number of Pareto solutions to present to user
    
    # Data Configuration
    DATA_SOURCE = "yfinance"
    CACHE_DURATION = 3600
    DEFAULT_SYMBOLS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        'NVDA', 'TSLA', 'JPM', 'UNH', 'PG',
        'XOM', 'HD', 'V', 'MA', 'KO'
    ]
    DEFAULT_TICKERS = DEFAULT_SYMBOLS
    MAX_STOCKS = 20
    DEFAULT_WINDOW_SIZE = 30
    
    # Default dates
    DEFAULT_START_DATE = "2022-01-01"
    DEFAULT_END_DATE = "2024-01-01"
    
    # Default weights for scalarization
    DEFAULT_WEIGHTS = {
        "return": 0.4,
        "risk": 0.3,
        "liquidity": 0.3
    }
    
    # Ideal point for Tchebycheff method
    DEFAULT_IDEAL_POINT = {
        "return": 1.0,
        "risk": 0.0,
        "liquidity": 1.0
    }
    
    # Flask Configuration - Read from environment variables
    SECRET_KEY = os.getenv('SECRET_KEY', 'fallback-secret-key-for-development')
    HOST = "127.0.0.1"
    PORT = 5000
    DEBUG = True
