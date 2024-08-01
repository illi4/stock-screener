## Stock screener

NASDAQ / ASX stocks bullish screener. 

Stock shortlisting logic uses the following conditions depending on the method:

**MRI** (method=mri) [NOTE: I do not currently use this method]
- Daily higher candle closes above bodies of the previous 5 daily candles and is green
- Bullish [MRI](https://tonevays.com/indicator) indicator value on the daily timeframe
- Bullish MRI indicator on the weekly timeframe  
- Moving averages (10, 20, 30 day) Consensio ([Guppy MMA](https://www.investopedia.com/terms/g/guppy-multiple-moving-average.asp))
- Moving averages (10, 20, 30 day) rising
- Market is not below MA200 with MA10 declining
- 2 most recent weekly candles closing above weekly moving averages (10/20/30 week)
- Close for the last week is not exceeding 300% when compared to 4 weeks ago
- Volume is significant in the last day compared to the 20-day moving average (defined in the config)  
- The stock has a significant range of movement over the past few weeks (defined in the config)
- Stochastic RSI is not overextended (>90%)

**Alphanumetrix** (method=anx)
- Price above MA200 
- Cross of MA7 and MA30
- Close for the last week is not exceeding 300% when compared to 4 weeks ago
- Note: Bullish SAR on weekly scale have to be checked manually as I cannot figure out the correct Lucid SAR calculation

For ASX, the best time to run it is in the evening after market closure to prepare for the next day. For US, that have to be morning (running in Australia) as US stocks data would refresh after a night of trading (per AU time). The shortlist acts as a guide for entering breakout trades which have high probability of success. As a data source, eodhistoricaldata.com is used.

**Please note that the shortlist should only be used to guide your own research and should not be interpreted as 'signals'. If you buy the stocks just using the output, you will definitely lose all your money.** 

### Usage  
- Run `python scanner.py --h` to view context help 
- To update the stocks list, run `python scanner.py --update`. It is recommended to run this daily prior to scanning. You can use the `-date` parameter (the format is `YYYY-MM-DD`).   
- To scan and shortlist, run `python scanner.py --scan -method=anx`. 
  - Use `method` to specify the method used (`mri` or `anx`). 
  - Use `stocks` if you want to force check specific stocks only. Separate by colon. E.g. `-stocks=NVDA,MSFT`. 
  - Use `num` to  limit amount of stocks scanned. E.g. `-num=100`. 
- To simulate scanning as of a particular date, use the `-date` parameter (the format is `YYYY-MM-DD`). For example, `python scanner.py --update -date=2021-01-05`.
- Helper scripts (note: requires configuring Google credentials in order to work). Use `-method` parameter as this will influence which spreadsheet is used. 
   - `monitor.py` to run daily to check whether the exit condition was hit for active entries. E.g. `python monitor.py -method=anx`
   - `paperfill.py` to run daily to fill in the values for paper trade entries automatically. Also fills in values for a few technical indicators if not populated. 
   - `simulator.py` simulates outcomes per the spreadsheet. Covered in details below. 
  This argument is there because the rule on using market conditions was already integrated in methodology and used for the stock selection with scanner.  
- The monitor would notify: 
  - when the close for a position is below MA10 
  - when the market switches to the bearish mode (market below MA200 with MA10 decreasing) as a trigger to close all open positions

### Running Simulations

The `simulator.py` script allows you to simulate trading outcomes based on historical data and your configured strategies. Here's how to use it:

```bash
python simulator.py -start=YYYY-MM-DD -end=YYYY-MM-DD -method=anx [options]
```

Example: `python simulator.py -mode=main -start=2023-12-10 -end=2024-04-01 -method=mri --show_monthly`

#### Required Parameters:

- -start: Start date for the simulation (format: YYYY-MM-DD)
- -end: End date for the simulation (format: YYYY-MM-DD)
- -method: Method of shortlisting (choices: mri, anx)

#### Optional Parameters:

- --plot: Generate and include plots in the output Excel file
- --forced_price_update: Force update of price data in the database

#### Configuration Parameters (in config.yaml):
The simulator section in config.yaml contains important parameters for the simulation:

- capital: Initial capital for the simulation
- commission: Commission fee per trade
- simultaneous_positions: List of maximum simultaneous positions to simulate
- stop_loss_level: Stop loss level as a decimal (e.g., 0.1 for 10%)
- numerical_filters: Filters for technical indicators (e.g., Fisher, Coppock)
- take_profit_variants: Different take-profit strategies to simulate

Example take_profit_variant:
```
  take_profit_values:
    - level: 5%
      exit_proportion: 25%
      move_stop_price: False
    - level: 10%
      exit_proportion: 25%
      move_stop_price: True
      move_stop_from_tp_level: 5%
```

This variant has two take-profit levels. At 5% profit, it exits 25% of the position. At 10% profit, it exits another 25% and moves the stop loss to 5% below the 10% take-profit level.

#### Output
The simulation results are saved in an Excel file named sim_summary.xlsx, containing:

- Summary sheet with performance metrics for each variant
- Monthly breakdown of capital values
- Plots of capital over time (if --plot is used)


### Settings 
See `config.yaml` for settings which include:
- Market to use 
- Price range for stocks considered on scan
- Minimum volume threshold  
- Overextended threshold
- Other conditions and rules

Your eodhistoricaldata API key must be placed in `.envrc` under variable `API_KEY`

### Limitations
- The scripts were tested on a machine in the `Australia/Sydney` timezone.
- Python 3.6+ is required.

### Installation

1. Clone this repo: `git clone https://github.com/illi4/asx-screener.git`
2. Install Python 3 and virtualenv if you do not have it installed yet. The easiest way is to download and install from the [official website](https://www.python.org/downloads/). Instructions to install virtualenv are published on the [python packaging website](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). 
3. Change to the cloned folder, create a python3 venv and activate it. 
    
    Linux / Mac: 
    ```
    python3 -m venv venv
    . venv/bin/activate
    ```
   
4. Install the required libraries: `pip install -r requirements.txt` 
5. Run referring to the usage instructions. 
 

### Google sheet and Google project configuration 
If you want to log your stocks and use monitor and paperfill in addition to the screener, please create a sheet similar to [this one](https://docs.google.com/spreadsheets/d/12uNaLya_qiQbT4NDbTaaQr0Y2sDbfDmEZDhvlzTRyjc/edit?usp=sharing) and configure API access, then save the credentials under `.config\gspread\service_account.json`. 
This is not required if you only want to use the scanner (scanner.py). 

There is a warning thrown when using gsheets, which is a [known issue](https://github.com/burnash/gspread/issues/1348) and doesn't affect the outcomes. 

**Note**: legacy sheet (R&D 2021) covering 2021 and before is available [here](https://docs.google.com/spreadsheets/d/1luuTn-wRsa2IXkaLTB-3FGlev6gJy6fnO0uQfqnHjRI/edit?usp=sharing).
