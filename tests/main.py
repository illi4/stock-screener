import os, sys
import string

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # to access the parent dir

from libs.stocktools import get_stock_data

all_letters = list(string.ascii_lowercase)


for extras in [1, 2, 5]:
    all_letters.append(extras)


print(all_letters)

#ohlc_daily, volume_daily = get_stock_data("TSLA")
#print(ohlc_daily)
