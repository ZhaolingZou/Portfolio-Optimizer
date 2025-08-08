# backend/portfolio_engine.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

from data_loader import download_data
from rolling_strategy import single_day_optimizer, rolling_portfolio_optimizer
from preference_extractor import PreferenceExtractor
from pareto_optimizer import generate_pareto_solutions, select_diverse_solutions
from config import Config

class PortfolioEngine:
    def __init__(self, tickers: List[str] = None):
        """
        Initialize the portfolio optimization engine.
        
        Parameters:
            tickers (list): List of stock tickers to include in portfolio
        """
        self.tickers = tickers or Config.DEFAULT_TICKERS
        self.prices = None
        self.volumes = None
        self.data_loaded = False
        self.preference_extractor = PreferenceExtractor()
        self.current_pareto_solutions = None
        self.user_session_data = {}
        
    def load_data(self, start_date: str = None, end_date: str = None) -> bool:
        """
        Load historical price and volume data for the specified tickers.
        
        Parameters:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            bool: True if data loaded successfully
        """
        try:
            start_date = start_date or Config.DEFAULT_START_DATE
            end_date = end_date or Config.DEFAULT_END_DATE
            
            print(f"Loading data for {len(self.tickers)} tickers from {start_date} to {end_date}...")
            
            self.prices, self.volumes = download_data(self.tickers, start_date, end_date)
            
            # Remove tickers with insufficient data
            min_data_points = Config.DEFAULT_WINDOW_SIZE + 10
            valid_tickers = []
            
            for ticker in self.tickers:
                if ticker in self.prices.columns:
                    if self.prices[ticker].notna().sum() >= min_data_points:
                        valid_tickers.append(ticker)
                    else:
                        print(f"Warning: {ticker} has insufficient data, removing from portfolio")
                else:
                    print(f"Warning: {ticker} data not available, removing from portfolio")
            
            self.tickers = valid_tickers
            self.prices = self.prices[self.tickers]
            self.volumes = self.volumes[self.tickers]
            
            self.data_loaded = len(self.tickers) > 0
            print(f"Data loaded successfully for {len(self.tickers)} tickers")
            
            return self.data_loaded
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data_loaded = False
            return False

    def generate_pareto_frontier(self, target_date: str = None) -> Dict[str, Any]:
        """
        Generate Pareto optimal solutions for user selection.
        
        Parameters:
            target_date (str): Target date for optimization (default: latest available)
            
        Returns:
            dict: Dictionary containing Pareto solutions and metadata
        """
        if not self.data_loaded:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        try:
            # Use latest available date if not specified
            if target_date is None:
                target_date = self.prices.index[-Config.DEFAULT_WINDOW_SIZE-1].strftime('%Y-%m-%d')
            
            target_date = pd.to_datetime(target_date)
            t_index = self.prices.index.get_loc(target_date)
            
            if t_index < Config.DEFAULT_WINDOW_SIZE:
                raise ValueError("Insufficient historical data for optimization")
            
            # Get historical window for calculations
            window_prices = self.prices.iloc[t_index - Config.DEFAULT_WINDOW_SIZE:t_index]
            window_volumes = self.volumes.iloc[t_index - Config.DEFAULT_WINDOW_SIZE:t_index]
            
            # Calculate expected returns
            daily_returns = window_prices.pct_change().dropna()
            expected_returns = daily_returns.mean()
            
            # Calculate covariance matrix
            cov_matrix = daily_returns.cov()
            
            # Calculate liquidity scores
            avg_volume = window_volumes.mean()
            liquidity_scores = (avg_volume - avg_volume.min()) / (avg_volume.max() - avg_volume.min() + 1e-8)
            
            # Generate Pareto optimal solutions
            print("Generating Pareto optimal solutions...")
            pareto_solutions = generate_pareto_solutions(expected_returns, cov_matrix, liquidity_scores)
            
            # Select diverse solutions for presentation
            selected_solutions = select_diverse_solutions(pareto_solutions)
            
            # Store current solutions for later reference
            self.current_pareto_solutions = selected_solutions
            
            return {
                "success": True,
                "pareto_solutions": selected_solutions,
                "n_solutions": len(selected_solutions),
                "optimization_date": target_date.strftime('%Y-%m-%d'),
                "message": f"Generated {len(selected_solutions)} diverse Pareto optimal portfolios for your selection."
            }
            
        except Exception as e:
            print(f"Error generating Pareto frontier: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to generate Pareto optimal solutions."
            }
    
    def extract_preferences_from_selection(self, selected_solution_id: int) -> Dict[str, float]:
        """
        Extract user preferences based on their selection from Pareto solutions.
        
        Parameters:
            selected_solution_id (int): ID of the selected Pareto solution
            
        Returns:
            dict: Extracted preference weights
        """
        if not self.current_pareto_solutions:
            raise ValueError("No Pareto solutions available. Generate Pareto frontier first.")
        
        # Find the selected solution
        selected_solution = None
        for solution in self.current_pareto_solutions:
            if solution['id'] == selected_solution_id:
                selected_solution = solution
                break
        
        if not selected_solution:
            raise ValueError(f"Solution with ID {selected_solution_id} not found.")
        
        # Extract preferences
        objectives = selected_solution['objectives']


        returns_arr     = np.array([s['objectives']['return']     for s in self.current_pareto_solutions], dtype=float)
        risks_arr       = np.array([s['objectives']['risk']       for s in self.current_pareto_solutions], dtype=float)
        liquidities_arr = np.array([s['objectives']['liquidity']  for s in self.current_pareto_solutions], dtype=float)


        eps = 1e-12
        def _score(val, arr, invert=False):
            if invert:
                val = arr.max() - val
                arr = arr.max() - arr
            rng = max(arr.max() - arr.min(), eps)
            return (val - arr.min()) / rng

        score_ret  = _score(objectives['return'],     returns_arr)            
        score_risk = _score(objectives['risk'],       risks_arr, invert=True) 
        score_liq  = _score(objectives['liquidity'],  liquidities_arr)        

        scores = np.array([score_ret, score_risk, score_liq], dtype=float)

        min_w = 0.10        
        n_obj = 3

        if scores.sum() == 0:
            preferences = Config.DEFAULT_WEIGHTS.copy()
        else:
            w_raw = scores / scores.sum()             
            residual = max(1.0 - n_obj * min_w, 0.0)  
            w_adj = w_raw * residual + min_w          
            preferences = {
                "return":     float(w_adj[0]),
                "risk":       float(w_adj[1]),
                "liquidity":  float(w_adj[2])
            }

        return preferences
    
    def optimize_portfolio(self, preferences: Dict[str, float], target_date: str = None, 
                          method: str = "tchebycheff") -> Dict[str, Any]:
        """
        Optimize portfolio based on user preferences.
        
        Parameters:
            preferences (dict): User preference weights
            target_date (str): Target date for optimization (default: latest available)
            method (str): Optimization method ('weighted_sum' or 'tchebycheff')
            
        Returns:
            dict: Optimization results including weights and metrics
        """
        if not self.data_loaded:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        try:
            # Use latest available date if not specified
            if target_date is None:
                target_date = self.prices.index[-Config.DEFAULT_WINDOW_SIZE-1].strftime('%Y-%m-%d')
            
            # Prepare ideal point for Tchebycheff method
            ideal_point = Config.DEFAULT_IDEAL_POINT if method == "tchebycheff" else None
            
            # Optimize portfolio
            weights = single_day_optimizer(
                prices=self.prices,
                volumes=self.volumes,
                target_date=target_date,
                window_size=Config.DEFAULT_WINDOW_SIZE,
                method=method,
                scalar_weights=preferences,
                ideal_point=ideal_point
            )
            
            # Calculate portfolio metrics
            metrics = self._calculate_portfolio_metrics(weights, target_date)
            
            # Filter out very small weights for cleaner presentation
            filtered_weights = {k: v for k, v in weights.items() if v > 0.01}
            
            # Normalize filtered weights to sum to 1
            total_weight = sum(filtered_weights.values())
            if total_weight > 0:
                filtered_weights = {k: v/total_weight for k, v in filtered_weights.items()}
            
            return {
                "weights": filtered_weights,
                "all_weights": weights,
                "metrics": metrics,
                "preferences_used": preferences,
                "optimization_date": target_date,
                "method": method,
                "tickers_included": self.tickers
            }
            
        except Exception as e:
            print(f"Error in portfolio optimization: {e}")
            raise
    
    def _calculate_portfolio_metrics(self, weights: Dict[str, float], target_date: str) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.
        
        Parameters:
            weights (dict): Portfolio weights
            target_date (str): Target date
            
        Returns:
            dict: Portfolio metrics
        """
        try:
            target_date = pd.to_datetime(target_date)
            t_index = self.prices.index.get_loc(target_date)
            
            if t_index < Config.DEFAULT_WINDOW_SIZE:
                raise ValueError("Insufficient historical data for metrics calculation")
            
            # Get historical window
            window_prices = self.prices.iloc[t_index - Config.DEFAULT_WINDOW_SIZE:t_index]
            window_volumes = self.volumes.iloc[t_index - Config.DEFAULT_WINDOW_SIZE:t_index]
            
            # Calculate returns
            daily_returns = window_prices.pct_change().dropna()
            
            # Portfolio metrics calculation
            portfolio_weights = np.array([weights.get(ticker, 0.0) for ticker in self.tickers])
            
            # Expected return
            expected_returns = daily_returns.mean()
            portfolio_return = np.dot(portfolio_weights, expected_returns)
            
            # Risk (volatility)
            cov_matrix = daily_returns.cov()
            portfolio_variance = np.dot(portfolio_weights, np.dot(cov_matrix, portfolio_weights))
            portfolio_risk = np.sqrt(portfolio_variance)
            
            # Liquidity
            avg_volumes = window_volumes.mean()
            normalized_liquidity = (avg_volumes - avg_volumes.min()) / (avg_volumes.max() - avg_volumes.min() + 1e-8)
            portfolio_liquidity = np.dot(portfolio_weights, normalized_liquidity)
            
            # Sharpe ratio
            sharpe_ratio = (portfolio_return - Config.RISK_FREE_RATE) / portfolio_risk if portfolio_risk > 0 else 0
            
            return {
                'return': float(portfolio_return),
                'risk': float(portfolio_risk),
                'liquidity': float(portfolio_liquidity),
                'sharpe_ratio': float(sharpe_ratio)
            }
            
        except Exception as e:
            print(f"Error calculating portfolio metrics: {e}")
            return {
                'return': 0.0,
                'risk': 0.0,
                'liquidity': 0.0,
                'sharpe_ratio': 0.0
            }
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about loaded data."""
        if not self.data_loaded:
            return {"error": "No data loaded"}
        
        return {
            "tickers": self.tickers,
            "n_tickers": len(self.tickers),
            "date_range": {
                "start": self.prices.index[0].strftime('%Y-%m-%d'),
                "end": self.prices.index[-1].strftime('%Y-%m-%d')
            },
            "data_points": len(self.prices)
        }

    
    def process_user_feedback(self, user_input: str, current_preferences: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Process user feedback and extract new preferences.
        
        Parameters:
            user_input (str): User's natural language feedback
            current_preferences (dict): Current preference weights
            
        Returns:
            dict: Processed feedback with new preferences and suggested weights
        """
        current_preferences = current_preferences or Config.DEFAULT_WEIGHTS
        
        # Extract preferences using GPT
        preference_result = self.preference_extractor.extract_preferences(
            user_input, current_preferences
        )
        
        # Add suggested_weights key that frontend expects
        if preference_result["success"]:
            preference_result["suggested_weights"] = preference_result["preferences"]
        else:
            preference_result["suggested_weights"] = current_preferences
        
        return preference_result
    
    def get_portfolio_explanation(self, optimization_result: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of optimization results.
        
        Parameters:
            optimization_result (dict): Results from optimize_portfolio()
            
        Returns:
            str: Human-readable explanation
        """
        return self.preference_extractor.generate_explanation(
            optimization_result["weights"],
            optimization_result["metrics"]
        )
    
    def get_available_tickers(self) -> List[str]:
        """Get list of available tickers."""
        return self.tickers.copy()
    
