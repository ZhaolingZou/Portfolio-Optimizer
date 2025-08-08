import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scalarization import weighted_sum_scalarization, tchebycheff_scalarization

def optimize_portfolio(expected_returns, cov_matrix, liquidity_scores,
                       method="weighted_sum", weights=None, ideal_point=None):
    """
    Solve a single-period portfolio optimization problem using scalarization.

    Parameters:
        expected_returns (pd.Series): Expected returns for each asset
        cov_matrix (pd.DataFrame): Covariance matrix of returns
        liquidity_scores (pd.Series): Normalized liquidity scores
        method (str): Scalarization method ('weighted_sum' or 'tchebycheff')
        weights (dict): Scalarization weights (e.g., {'return': 1.0, 'risk': -0.5, 'liquidity': 0.2})
        ideal_point (dict): Ideal values for objectives (only needed for 'tchebycheff')

    Returns:
        dict: Optimized portfolio weights (key = ticker, value = weight)
    """
    tickers = expected_returns.index.tolist()
    n = len(tickers)

    # Define objective function to minimize (negative because we maximize score)
    def objective(w):
        port_return = np.dot(w, expected_returns)
        port_risk = np.dot(w, np.dot(cov_matrix, w)) ** 0.5
        port_liquidity = np.dot(w, liquidity_scores)

        obj_values = {'return': port_return, 'risk': port_risk, 'liquidity': port_liquidity}

        if method == "weighted_sum":
            score = weighted_sum_scalarization(obj_values, weights)
            return -score
        elif method == "tchebycheff":
            tcheby_score = tchebycheff_scalarization(obj_values, weights, ideal_point)
            return tcheby_score
    


    # Constraints: weights sum to 1, each weight ∈ [0,1]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(n)]
    init_guess = np.ones(n) / n  # equal weights as initial guess

    result = minimize(objective, init_guess, bounds=bounds, constraints=constraints)

    if not result.success:
        raise ValueError("Optimization failed:", result.message)

    optimized_weights = {tickers[i]: result.x[i] for i in range(n)}
    return optimized_weights

def optimize_scalarized_objective(scalarized_scores):
    """
    Optimize portfolio weights based only on scalarized asset scores.

    Parameters:
        scalarized_scores (np.array): A 1D array of scalarized scores per asset.

    Returns:
        list: Optimal portfolio weights.
    """
    n_assets = len(scalarized_scores)

    def objective(w):
        return -np.dot(w, scalarized_scores)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(n_assets)]
    init_guess = np.ones(n_assets) / n_assets

    result = minimize(objective, init_guess, bounds=bounds, constraints=constraints)

    if not result.success:
        raise ValueError("Optimization failed:", result.message)

    return result.x.tolist()