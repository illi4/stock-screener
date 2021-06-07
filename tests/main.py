import os, sys
import arrow

# TEST
'''
current_date = arrow.now()
shifted_date = current_date.shift(years=-1)
print(shifted_date.format("YYYY-MM-DD"))

exit(0)
'''
# TEST

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # to access the parent dir

from libs.stocktools import get_stock_data

ohlc_daily, volume_daily = get_stock_data("CVS.AU")
print(ohlc_daily)
print(ohlc_daily.dtypes)
