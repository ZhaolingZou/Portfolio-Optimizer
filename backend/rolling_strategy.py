import pandas as pd
import numpy as np
from optimizer import optimize_portfolio
from scalarization import weighted_sum_scalarization, tchebycheff_scalarization

def compute_liquidity_scores(volume_window):
    """
    Normalize average daily volume over a window to a 0-1 liquidity score.
    """
    avg_volume = volume_window.mean()
    norm_scores = (avg_volume - avg_volume.min()) / (avg_volume.max() - avg_volume.min() + 1e-8)
    return norm_scores

def normalize_objectives(mu, sigma, liquidity, daily_returns):
    """
    Normalize return, risk (inverted), and liquidity to 0–1.
    Also returns the normalized and inverted covariance matrix.
    
    Parameters:
        mu (pd.Series): Expected returns
        sigma (pd.Series): Standard deviations
        liquidity (pd.Series): Liquidity scores
        daily_returns (pd.DataFrame): Raw daily return data used to compute correlation
    """
    mu_scaled = (mu - mu.min()) / (mu.max() - mu.min() + 1e-8)

    # Compute correlation matrix from raw return data
    corr_matrix = daily_returns.corr()
    
    # Normalize standard deviation
    sigma_scaled = (sigma - sigma.min()) / (sigma.max() - sigma.min() + 1e-8)
    
    # Reconstruct normalized covariance matrix: Cov = Corr × σ_i × σ_j
    sigma_outer = np.outer(sigma_scaled, sigma_scaled)
    cov_scaled = corr_matrix.values * sigma_outer
    cov_scaled = pd.DataFrame(cov_scaled, index=sigma.index, columns=sigma.index)
    
    # Invert covariance matrix: lower risk → higher score
    # Use maximum eigenvalue to ensure positive definiteness after inversion
    max_eigenvalue = np.max(np.linalg.eigvals(cov_scaled.values))
    cov_scaled_inverted = max_eigenvalue * np.eye(len(cov_scaled)) - cov_scaled.values
    cov_scaled_inverted = pd.DataFrame(cov_scaled_inverted, index=cov_scaled.index, columns=cov_scaled.columns)
    
    # Ensure matrix remains positive definite
    cov_scaled_inverted += 1e-6 * np.eye(len(cov_scaled_inverted))
    
    # Risk score: lower risk → higher score
    sigma_score = 1 - sigma_scaled
    liquidity_scaled = liquidity  # Already in [0, 1]

    return mu_scaled, sigma_score, liquidity_scaled, cov_scaled_inverted


def rolling_portfolio_optimizer(prices, volumes, window_size=30, method="weighted_sum",
                                scalar_weights=None, ideal_point=None):
    """
    Perform rolling daily portfolio optimization using normalized objectives.
    """
    tickers = prices.columns.tolist()
    n_assets = len(tickers)
    weights_history = []

    for t in range(window_size, len(prices)):
        window_prices = prices.iloc[t-window_size:t]
        window_volumes = volumes.iloc[t-window_size:t]
        date = prices.index[t]

        # Estimate raw statistics
        daily_returns = window_prices.pct_change().dropna()
        mu = daily_returns.mean()
        sigma = daily_returns.std()
        cov = daily_returns.cov()
        liquidity = compute_liquidity_scores(window_volumes)

        # Normalize objectives (pass daily_returns to compute correlation)
        mu_scaled, sigma_scaled, liquidity_scaled, cov_scaled = normalize_objectives(
            mu, sigma, liquidity, daily_returns
        )

        # Optimize using normalized inputs
        weights = optimize_portfolio(
            expected_returns=mu_scaled,
            cov_matrix=cov_scaled,
            liquidity_scores=liquidity_scaled,
            method=method,
            weights=scalar_weights,
            ideal_point=ideal_point
        )
        weights['date'] = date
        weights_history.append(weights)

    weights_df = pd.DataFrame(weights_history).set_index('date')
    weights_df = weights_df[tickers]  # Ensure column order
    return weights_df


def single_day_optimizer(prices, volumes, target_date, window_size=30,
                         method="tchebycheff", scalar_weights=None, ideal_point=None):
    """
    Compute optimized portfolio weights for a single day using normalized objectives.
    """
    target_date = pd.to_datetime(target_date)

    if target_date not in prices.index:
        raise ValueError(f"Target date {target_date} not found in data.")

    t_index = prices.index.get_loc(target_date)
    if t_index < window_size:
        raise ValueError("Not enough historical data before the target date.")

    # Extract past window data
    window_prices = prices.iloc[t_index - window_size:t_index]
    window_volumes = volumes.iloc[t_index - window_size:t_index]

    # Estimate daily returns and raw statistics
    daily_returns = window_prices.pct_change().dropna()
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    cov = daily_returns.cov()
    liquidity = compute_liquidity_scores(window_volumes)

    # Normalize objectives (pass daily_returns)
    mu_scaled, sigma_scaled, liquidity_scaled, cov_scaled = normalize_objectives(
        mu, sigma, liquidity, daily_returns
    )

    # Optimize using normalized objectives
    weights = optimize_portfolio(
        expected_returns=mu_scaled,
        cov_matrix=cov_scaled,
        liquidity_scores=liquidity_scaled,
        method=method,
        weights=scalar_weights,
        ideal_point=ideal_point
    )

    return weights
