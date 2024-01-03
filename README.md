### Stock screener

NASDAQ / ASX stocks ticker grabber and a basic screener. **For ASX, the best time to run it is in the evening after market closure to prepare for the next day**.  
For US, that have to be morning (running in Australia) as US stocks data would refresh after a night of trading (per AU time). The shortlist acts as a guide for entering breakout trades which have high probability of success. As a data source, eodhistoricaldata.com is used.  

Stock shortlisting and industry score estimates incorporate the following conditions: 
- Daily higher candle closes above bodies of the previous 5 daily candles and is green
- Bullish [MRI](https://tonevays.com/indicator) indicator value on the daily timeframe
- Bullish MRI indicator on the weekly timeframe  
- Moving averages (10, 20, 30 day) Consensio ([Guppy MMA](https://www.investopedia.com/terms/g/guppy-multiple-moving-average.asp))
- Moving averages (10, 20, 30 day) rising
- Market is not below MA200 with MA10 declining
- 2 most recent weekly candles closing above weekly moving averages (10/20/30 week)
- Close for the last week is not exceeding 500% when compared to 4 weeks ago
- Volume is significant in the last day compared to the 20-day moving average (defined in the config)  

**Please note that the shortlist should only be used to guide your own research and should not be interpreted as 'signals'.** 

#### Usage  
- Run `python scanner.py --h` to view context help 
- To update the stocks list, run `python scanner.py --update -exchange=nasdaq`. It is recommended to run this daily prior to scanning.  
- To scan and shortlist, run `python scanner.py --scan -exchange=nasdaq`. 
- To simulate scanning as of a particular date, use the `-date` parameter (the format is `YYYY-MM-DD`). For example, `python scanner.py --update -exchange=nasdaq -date=2021-01-05`.
- Helper scripts (note: requires configuring Google credentials in order to work, not updated to the new format yet):  
   - `monitor.py` to run daily to check whether the exit condition was hit for active entries.
   - `paperfill.py` to run daily to fill in the values for paper trade entries automatically. **Note**: NASDAQ hardcoded at the moment. 
   - `simulator.py` simulates outcomes per the spreadsheet. **Note**: NASDAQ hardcoded at the moment. 
   - `simulator_legacy.py` works with the older 21 R&D spreadsheet and also has an optional argument `--market` which would include market MA200/MA10 conditions when running simulation **in the tp mode**.  
- The monitor would notify: 
  - when the close for a position is below MA10 
  - when the market switches to the bearish mode (market below MA200 with MA10 decreasing) as a trigger to close all open positions

Example running a simulator: 
`python simulator.py -mode=main -exchange=nasdaq -start=2022-02-10 -end=2022-04-01 --show_monthly`

#### Settings 
See `config.yaml` for settings: 
- Price range for stocks considered on scan
- Minimum volume threshold  
- Overextended threshold

API key must be placed in `.envrc` under variable `API_KEY`

#### Limitations
- The scripts were tested on a machine in the `Australia/Sydney` timezone.
- Python 3.6+ is required.

#### Installation

1. Clone the repo: `git clone https://github.com/illi4/asx-screener.git`
2. Install Python 3 and virtualenv if you do not have it installed yet. The easiest way is to download and install from the [official website](https://www.python.org/downloads/). Instructions to install virtualenv are published on the [python packaging website](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). 
3. Change to the cloned folder, create a python3 venv and activate it. 
    
    Linux / Mac: 
    ```
    python3 -m venv venv
    . venv/bin/activate
    ```
   
4. Install the required libraries: `pip install -r requirements.txt` 

#### Google sheet / project configuration 
If you want to use monitor and paperfill in addition to the screener, please create a sheet similar to [this one](https://docs.google.com/spreadsheets/d/12uNaLya_qiQbT4NDbTaaQr0Y2sDbfDmEZDhvlzTRyjc/edit?usp=sharing) and configure API access, then save the credentials under `.config\gspread\service_account.json`. This is not required if you only want to use the screener (main.py).

There is a warning thrown when using gsheets, which is a [known issue](https://github.com/burnash/gspread/issues/1348) and doesn't affect the outcomes. 

**Note**: legacy sheet (R&D 2021) covering 2021 and before is available [here](https://docs.google.com/spreadsheets/d/1luuTn-wRsa2IXkaLTB-3FGlev6gJy6fnO0uQfqnHjRI/edit?usp=sharing).