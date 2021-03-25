import os, sys
import string

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # to access the parent dir

from libs.stocktools import get_stock_data

for elem in string.ascii_lowercase:
    print (elem)

#ohlc_daily, volume_daily = get_stock_data("TSLA")
#print(ohlc_daily)
