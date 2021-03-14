### ASX screener

ASX stocks ticker grabber and a basic screener. Beware that the **current day is excluded** from the analysis, even if you run the screener after the market closes (this is an unfortunate consequence of using Yahoo Finance as a data source). Thus, **the best time to run it is in the morning prior to market open**. The shortlist acts as a guide for entering breakout trades which have high probability of success.

Stock shortlisting and industry score estimates incorporate the following conditions: 
- Daily higher candle close
- Bullish [MRI](https://tonevays.com/indicator) indicator value on daily timeframe
- Bullish MRI indicator on weekly timeframe (not close to exhaustion)
- Moving averages (50, 200 day) Consensio ([Guppy MMA](https://www.investopedia.com/terms/g/guppy-multiple-moving-average.asp))
- Moving averages (50, 200 day) rising 
- Close for the last week is not more than 200% compared to 4 weeks ago
- Volume spike in the last 5 days compared to the 20-day moving average (for stocks only and not for industries)

Please note that the shortlist should only be used to guide your own research and should not be interpreted as 'signals'. 

#### Usage  
- Run `main.py --h` to view context help 
- To update the stocks list, run `main.py --update`. Recommended to run this daily prior to scanning.
- To scan and shortlist, run `main.py --scan`

#### Settings 
See `libs/settings.py` for settings: 
- URL for grabbing the listed codes and the price
- Price range for stocks considered on scan
- Overextended threshold

#### Limitations
- Yahoo Finance is used to get stock OHLC information. Theoretically, the API limit for YFinance is 2000 requests per hour per IP. However, I have launched it several times in a row producing more than 2000 requests and was not yet able to hit the limit. If you do, the script will just keep retrying with 1-minute intervals until cancelled.
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
