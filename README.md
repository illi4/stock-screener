### ASX screener

ASX stocks ticker grabber and a basic screener. Shortlisting uses the following rules: 
- Daily higher candle close
- Bullish [MRI](https://tonevays.com/indicator) indicator value on daily timeframe
- Bullish MRI indicator on weekly timeframe (not close to exhaustion)
- Volume spike in the last 5 days compared to the 20-day moving average
- Moving averages Consensio ([Guppy MMA](https://www.investopedia.com/terms/g/guppy-multiple-moving-average.asp))
- Moving averages (30, 50, 200 day) rising 
- Close for the last week is not more than 100% compared to 4 weeks ago

Please note that the shortlist should only be used to guide your own research and should not be interpreted as 'signals'. 

#### Installation

1. Clone the repo: `git clone https://github.com/illi4/asx-screener.git`
2. Change to this folder, create a python3 venv and activate it. 
    
    Linux: 
    ```
    sudo apt-get install python3-venv
    python3 -m venv venv
    . venv/bin/activate
    ```
    Windows: 
    ```
    python -m venv venv
    venv＼Scripts＼activate
    ```
   
3. Install the required libraries: `pip install -r requirements.txt`
4. Install ChromeDriver

#### Usage  
- Run `main.py --h` to view context help 
- To update the stocks list, run `main.py --update` 
- To scan and shortlist, run `main.py --scan`

#### Settings 
See `libs/settings.py` for settings: 
- URL for grabbing the listed codes and the price
- Price range for stocks considered on scan

#### Limitations
- Yahoo Finance is used to get stock OHLC information. The API limit for YFinance is 2000 requests per hour per IP. If you hit the limit, the script will keep retrying until cancelled. 