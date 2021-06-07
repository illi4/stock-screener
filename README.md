### Stock screener

ASX / NASDAQ stocks ticker grabber and a basic screener. Beware that the **current day is excluded** from the analysis, even if you run the screener after the market closes (this is an unfortunate consequence of using Yahoo Finance as a data source). Thus, **the best time to run it is in the morning prior to market open**. The shortlist acts as a guide for entering breakout trades which have high probability of success.

**Note that the YFinance data may be unreliable at times and it is crucial to double check the shortlist on the actual charts prior to any actions.**

Stock shortlisting and industry score estimates incorporate the following conditions: 
- Daily higher candle closes above bodies of the previous 10 daily candles and is green
- Bullish [MRI](https://tonevays.com/indicator) indicator value on the daily timeframe
- Bullish MRI indicator on the weekly timeframe  
- Moving averages (10, 20, 30 day) Consensio ([Guppy MMA](https://www.investopedia.com/terms/g/guppy-multiple-moving-average.asp))
- Moving averages (10, 20, 30 day) rising
- 2 most recent weekly candles closing above weekly moving averages (10/20/30 week)
- Close for the last week is not exceeding 200% when compared to 4 weeks ago
- Volume spike in the last day compared to the 20-day moving average  

Please note that the shortlist should only be used to guide your own research and should not be interpreted as 'signals'. 

#### Usage  
- Run `main.py --h` to view context help 
- To update the stocks list, run `main.py --update -exchange=all`. It is recommended to run this daily prior to scanning.  
- To scan and shortlist, run `main.py --scan -exchange=all`.
- Helper scripts (note: requires configuring Google credentials in order to work):  
   - `monitor.py` to run daily to check whether the exit condition was hit for active entries.
   - `paperfill.py` to run daily to fill in the values for paper trade entries automatically.

#### Settings 
See `libs/settings.py` for settings: 
- URL for grabbing the listed codes and the price
- Price range for stocks considered on scan
- Minimum volume threshold (500k)  
- Overextended threshold

#### Limitations
- Yahoo Finance is used to get stock OHLC information. Theoretically, the API limit for YFinance is 2000 requests per hour per IP. However, I have launched it several times in a row producing more than 2000 requests and was not yet able to hit the limit. If you do, the script will just keep retrying with 1-minute intervals until cancelled.
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
   
    Windows: 
    ```
    python -m venv venv
    venv＼Scripts＼activate
    ```
   
4. Install the required libraries: `pip install -r requirements.txt`
5. Install [ChromeDriver](https://sites.google.com/a/chromium.org/chromedriver/home)
6. Specify your eodhistoricaldata.com key in `.config\eoddata.json` using the following syntax: 
``` 
{
  "key": "XXXX.XXXXXXX"
}
```

#### Google sheet / project configuration 
If you want to use monitor and paperfill in addition to the screener, please create a sheet similar to [this one](https://docs.google.com/spreadsheets/d/1luuTn-wRsa2IXkaLTB-3FGlev6gJy6fnO0uQfqnHjRI/edit?usp=sharing) and configure API access, then save the credentials under `.config\gspread\service_account.json`. This is not required if you only want to use the screener (main.py).
