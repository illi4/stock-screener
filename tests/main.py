import os, sys
import arrow

current_datetime = arrow.now()
current_dow = current_datetime.isoweekday()
if current_dow == 1:  # only subtract if today is Monday
    current_datetime = current_datetime.shift(days=-3)
current_datetime = current_datetime.format("YYYY-MM-DD")

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

ohlc_daily, volume_daily = get_stock_data("VRAY")
print(ohlc_daily)
