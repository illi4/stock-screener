# Stock Screener and Trading Simulator

This project combines a stock screener for NASDAQ/ASX stocks with a trading simulator, allowing users to identify potential trades and test various trading strategies using historical data.

## Table of Contents
1. [Installation](#installation)
2. [Stock Screener](#stock-screener)
   - [Usage](#usage)
   - [Settings](#settings)
   - [Limitations](#limitations)
3. [Trading Simulator](#trading-simulator)
   - [Running Simulations](#running-simulations)
   - [Configuration](#configuration)
4. [Google Sheet Configuration](#google-sheet-and-google-project-configuration)
5. [Troubleshooting](#troubleshooting)

## Installation

1. Clone this repo: `git clone https://github.com/illi4/asx-screener.git`
2. Install Python 3 (3.6+) and virtualenv if not already installed.
3. Change to the cloned folder, create and activate a Python 3 venv:
   ```
   python3 -m venv venv
   . venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```
4. Install required libraries: `pip install -r requirements.txt`
5. Place your eodhistoricaldata API key in `.envrc` under variable `API_KEY`

## Stock Screener

The stock screener helps identify potential bullish trading opportunities based on various technical indicators and market data.

### Usage

Run `python scanner.py --h` to view context help

- Update stocks list: 
  ```
  python scanner.py --update [-date=YYYY-MM-DD]
  ```
- Scan and shortlist:
  ```
  python scanner.py --scan -method=anx [-stocks=STOCK1,STOCK2,...] [-num=100]
  ```

Helper scripts (requires Google credentials):

- Monitor exit conditions: 
    ```
    python monitor.py -method=anx
    ```
- Auto-fill paper trade entries: 
    ```
    python paperfill.py -method=anx
    ```
- **Daily routine**: 
    ```
    python paperfill.py -method=anx
    python monitor.py -method=anx
    python scanner.py --update
    python scanner.py --scan -method=anx
    ```

### Settings

See `config.yaml` for settings including:
- Market selection
- Price range for stocks
- Minimum volume threshold
- Overextended threshold
- Other conditions and rules

### Limitations

- Tested on Australia/Sydney timezone
- Requires Python 3.6+
- The shortlist should only guide your research, not be interpreted as 'signals'

## Trading Simulator

The trading simulator uses historical data to test various trading strategies and parameters.

### Running Simulations

To run the simulator:

```
python simulator.py -start=YYYY-MM-DD -end=YYYY-MM-DD -method=anx [options]
```

Example:
```
python simulator.py -start=2023-12-10 -end=2024-04-01 -method=mri --plot
```

#### Required Parameters:
- `-start=YYYY-MM-DD`: Start date for the simulation
- `-end=YYYY-MM-DD`: End date for the simulation
- `-method=anx|mri`: Method of shortlisting

#### Optional Parameters:
- `--plot`: Generate and include plots in the output Excel file
- `--forced_price_update`: Force update of price data in the database
- `-stock=STOCK_CODE`: Specify a single stock to simulate
- `--sampling`: Enable sampling mode for multiple simulation runs

### Configuration

The `config.yaml` file contains important simulation parameters:

- `capital`: Initial capital
- `commission`: Commission fee per trade
- `simultaneous_positions`: List of maximum simultaneous positions to simulate
- `numerical_filters`: Filters for technical indicators (e.g., Fisher, Coppock)
- `take_profit_variants`: Different take-profit strategies to simulate

Example take_profit_variant:
```yaml
take_profit_values:
  - level: 5%
    exit_proportion: 25%
    move_stop_price: False
  - level: 10%
    exit_proportion: 25%
    move_stop_price: True
    move_stop_from_tp_level: 5%
```

#### Output
Results are saved in `sim_summary.xlsx`, containing:
- Summary of performance metrics for each variant
- Monthly breakdown of capital values
- Plots of capital over time (if `--plot` is used)

## Google Sheet and Google Project Configuration

To log stocks and use monitor and paperfill features:
1. Create a sheet similar to [this example](https://docs.google.com/spreadsheets/d/12uNaLya_qiQbT4NDbTaaQr0Y2sDbfDmEZDhvlzTRyjc/edit?usp=sharing)
2. Configure API access
3. Save credentials under `.config\gspread\service_account.json`

Note: This is not required for using only the scanner (scanner.py)

## Troubleshooting

- There's a known warning when using gsheets, which doesn't affect outcomes. See [this issue](https://github.com/burnash/gspread/issues/1348)
- For other issues, ensure all required Python packages are installed and up to date
- Check that your Google Sheets is properly formatted and accessible if using related features

