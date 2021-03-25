import os, sys
import string

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # to access the parent dir

from libs.stocktools import get_stock_data

x = []

a = [1, 2, 3]
b = [3,4]

x.append(a)
x.append(b)
print(x)

#ohlc_daily, volume_daily = get_stock_data("TSLA")
#print(ohlc_daily)
