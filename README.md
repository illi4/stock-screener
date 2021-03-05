### ASX screener

ASX stocks ticker grabber and a basic screener. Stock shortlisting and industry 'bullishness' estimates use the following rules: 
- Daily higher candle close
- Bullish [MRI](https://tonevays.com/indicator) indicator value on daily timeframe
- Bullish MRI indicator on weekly timeframe (not close to exhaustion)
- Moving averages Consensio ([Guppy MMA](https://www.investopedia.com/terms/g/guppy-multiple-moving-average.asp))
- Moving averages (30, 50, 200 day) rising 
- Close for the last week is not more than 100% compared to 4 weeks ago
- Volume spike in the last 5 days compared to the 20-day moving average (for stocks only, not for industry)

Please note that the shortlist should only be used to guide your own research and should not be interpreted as 'signals'. 

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

#### Usage  
- Run `main.py --h` to view context help 
- To update the stocks list, run `main.py --update`. Recommended to run this daily prior to scanning.
- To scan and shortlist, run `main.py --scan`. Exemplary output: 

```
Calculating industry momentum scores...
Scanning 1266 stocks priced 0.01 from to 1
PLS [PLS Pilbara Minerals Ltd] (1/1266)
- MRI: daily [x] / weekly [x] | Consensio: [v] | MA rising: [x] | Not overextended: [v] | Higher close: [x] | Volume condition: [v]
- [x] not meeting shortlisting conditions
...
- MRI: daily [x] / weekly [v] | Consensio: [x] | MA rising: [x] | Not overextended: [v] | Higher close: [x] | Volume condition: [v]
- [x] not meeting shortlisting conditions
Getting industry data for the shortlisted stocks...

All shortlisted stocks (sorted by 5-day moving average volume):
- BRK (BRK Brookside Energy Ltd) (Energy) [40.45M] | Industry bullishness score 2.9/5.0
- EPM (EPM Eclipse Metals Ltd) (Basic Materials) [8.12M] | Industry bullishness score 3.6/5.0
- FSG (FSG Field Solutions Holdings Ltd) (Communication Services) [1.12M] | Industry bullishness score 2.9/5.0
- RXP (RXP RXP Services Ltd) (Technology) [667.21K] | Industry bullishness score 4.3/5.0
- ZEU (ZEU ZEUS Resources Ltd) (Energy) [628.83K] | Industry bullishness score 2.9/5.0
- SRY (SRY Story-I Ltd) (Consumer Cyclical) [540.99K] | Industry bullishness score 2.9/5.0
- AVG (AVG Australian Vintage Ltd) (Consumer Defensive) [458.91K] | Industry bullishness score 2.1/5.0
- KGL (KGL KGL Resources Ltd) (Basic Materials) [266.45K] | Industry bullishness score 3.6/5.0
- TOU (TOU Tlou Energy Ltd) (Energy) [228.54K] | Industry bullishness score 2.9/5.0
- SHJ (SHJ Shine Justice Ltd) (Consumer Cyclical) [144.06K] | Industry bullishness score 2.9/5.0
- SKS (SKS SKS Technologies Group Ltd) (Industrials) [43.68K] | Industry bullishness score 1.4/5.0
- N1H (N1H N1 Holdings Ltd) (Financial Services) [13.53K] | Industry bullishness score 2.1/5.0
- AVC (AVC Auctus Investment Group Ltd) (Financial Services) [12.61K] | Industry bullishness score 2.1/5.0
```

#### Settings 
See `libs/settings.py` for settings: 
- URL for grabbing the listed codes and the price
- Price range for stocks considered on scan

#### Limitations
- Yahoo Finance is used to get stock OHLC information. The API limit for YFinance is 2000 requests per hour per IP. If you hit the limit, the script will keep retrying until cancelled.
- Written using Python 3.8 and not tested with other versions. 