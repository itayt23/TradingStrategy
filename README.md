# Trading Strategy Repository

## Overview

This repository contains a generic trading strategy implementation that allows users to test various strategies in the stock market using a backtester. Additionally, it includes an optimizer function to explore different rule combinations set by the user.

## Knowledge Used

The trading strategy leverages various technical indicators and market conditions to make buy and sell decisions. Some of the key indicators and conditions include:

- Simple Moving Averages (SMA)
- Moving Average Convergence Divergence (MACD)
- Relative Strength Index (RSI)
- Volatility Index (VIX)
- Trend conditions
- Candlestick patterns (e.g., daily_below_lower_band, cc, dew)
- Historical price changes (e.g., last_year_change)

## Usage Example

To use this trading strategy, follow these steps:

1. **Set Up Strategy Rules:**
   - Use the `add_rule_buy_sell` method to define buy and sell conditions based on your preferred indicators and thresholds.

   ```python
   trading_strategy = TradingStrategy(verbose=False)
   trading_strategy.add_rule_buy_sell(lambda row: row['sma10'] < row['sma5'] and row['vix_sma10'] > row['vix_sma5'], type='buy') 
   # Add more rules based on your requirements...
    ```


2. **Set Up Benchmark and Run Backtest:**
   - Specify a benchmark symbol (e.g., 'SPY') and run the backtest.

   ```python
    trading_strategy.benchmark = ['SPY']
    trading_strategy.build_dataframe()
    trading_strategy.run_backtest(start_date='2018-01-01', verbose=True)
    ```


3. **Optimize Strategy Rules:**
   - Use the `optimizer` function to explore different rule combinations and optimize the strategy.

   ```python
    trading_strategy.optimizer(start_date='2018-01-01', optimize_weight=True, optimize_val=True, estimated_runtime=True)
     ```

## Notes

- Additional conditions, thresholds, and parameters can be customized based on user preferences.
- To save the backtest results as a graph, for that ensure `verbose` is set to `True` in the `run_backtest` function to enable graph saving.


## Setup Instructions

1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt` (if applicable).
3. Follow the usage example and steps mentioned in the README.md file.