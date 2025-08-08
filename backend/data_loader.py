import yfinance as yf
import pandas as pd

def download_data(tickers, start_date, end_date):
    """
    Download adjusted closing prices and volume data from Yahoo Finance.

    Parameters:
        tickers (list of str): List of stock ticker symbols (e.g., ['AAPL', 'MSFT'])
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
        adj_close (pd.DataFrame): Adjusted closing prices
        volume (pd.DataFrame): Daily trading volume
    """
    # Download historical data for multiple tickers
    # The result is a DataFrame with MultiIndex columns: (Ticker, Field)
    raw_data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        group_by='ticker',
        auto_adjust=False,
        progress=False  # Disable progress bar for cleaner output
    )

    # Extract adjusted closing prices using the second-level column 'Adj Close'
    adj_close = raw_data.xs('Adj Close', axis=1, level=1)

    # Extract volume data
    volume = raw_data.xs('Volume', axis=1, level=1)

    # Drop any rows with missing data and align volume with available dates
    adj_close.dropna(inplace=True)
    volume = volume.loc[adj_close.index]

    return adj_close, volume

