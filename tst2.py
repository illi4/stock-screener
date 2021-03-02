'''
name_list = ['Joe Smoe', 'Jane Doe']
gender_list = ['male', 'female']
age_list = [34, 21]
city_list = ['Joe City', 'Jane City']
state_list = ['Joe State', 'Jane State']

completed_dict = [{
                 'name': a, 'gender': b, 'age': c, 'city': d, 'state': e}
                 for a, b, c, d, e, in zip(name_list, gender_list,  age_list, city_list, state_list)]

print(completed_dict)
'''

# This is ok
from libs.db import bulk_add_stocks, create_stock_table, delete_all_stocks

create_stock_table()
delete_all_stocks()
stocks = [{'code': 'AAA', 'name': 'CBA Commonwealth Bank of Australia', 'price': 11.49},
          {'code': 'XXX', 'name': 'BHP BHP Group Ltd', 'price': 49.13},
          {'code': 'CSL', 'name': 'CSL CSL Ltd', 'price': 269.74},
          {'code': 'NAB', 'name': 'NAB National Australia Bank Ltd', 'price': 25.26}]
bulk_add_stocks(stocks)