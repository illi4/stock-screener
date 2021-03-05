### ASX screener

ASX stocks ticker grabber and a basic screener. Beware that the **current day is excluded** from the analysis, even if you run the screener after the market closes. 

Stock shortlisting and industry 'bullishness' estimates use the following rules: 
- Daily higher candle close
- Bullish [MRI](https://tonevays.com/indicator) indicator value on daily timeframe
- Bullish MRI indicator on weekly timeframe (not close to exhaustion)
- Moving averages Consensio ([Guppy MMA](https://www.investopedia.com/terms/g/guppy-multiple-moving-average.asp))
- Moving averages (30, 50, 200 day) rising 
- Close for the last week is not more than 100% compared to 4 weeks ago
- Volume spike in the last 5 days compared to the 20-day moving average (for stocks only, not for industry)

Please note that the shortlist should only be used to guide your own research and should not be interpreted as 'signals'. 

#### Usage  
- Run `main.py --h` to view context help 
- To update the stocks list, run `main.py --update`. Recommended to run this daily prior to scanning.
- To scan and shortlist, run `main.py --scan`

#### Settings 
See `libs/settings.py` for settings: 
- URL for grabbing the listed codes and the price
- Price range for stocks considered on scan

#### Limitations
- Yahoo Finance is used to get stock OHLC information. The API limit for YFinance is 2000 requests per hour per IP. If you hit the limit, the script will keep retrying until cancelled.
- Written using Python 3.8 and not tested with other versions. 

#### Installation

1. Clone the repo: `git clone https://github.com/illi4/asx-screener.git`
2. Install Python 3 and virtualenv if you do not have it installed yet. 

    Mac: 
    ```
    brew install python3
    pip3 install virtualenv
    ```
   
   Windows: download and install from the [official website](https://www.python.org/downloads/). 

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
