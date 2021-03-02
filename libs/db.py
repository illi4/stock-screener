from peewee import *
import peewee
import datetime

db = SqliteDatabase("stocks.db")

class BaseModel(Model):
    class Meta:
        database = db


class Stock(BaseModel):
    code = CharField()
    name = CharField()
    price = FloatField(null=True)
    date = DateTimeField(default=datetime.datetime.now)


def create_stock_table():
    Stock.create_table()


def delete_all_stocks():
    query = Stock.delete()
    query.execute()


def bulk_add_stocks(stocks_dict):
    with db.atomic():
        Stock.insert_many(stocks_dict).execute()


def get_stocks(price_min=None, price_max=None):
    price_min = 0 if price_min is None else price_min
    price_max = 10e9 if price_max is None else price_max

    try:
        stocks = (
            Stock.select()
                .where(
                (Stock.price >= price_min)
                & (Stock.price < price_max)
            )
        )

        '''
        stocks = (
            Stock.select()
            .where(
                (Trade.entry_date_us >= start_date)
                & (Trade.entry_date_us == Trade.exit_date_us)
            )
            .count()
        )
        '''
        if len(stocks) == 0:
            print('Warning: no stocks in the database')
    except peewee.OperationalError:
        print('Error: table not found. Update the stocks list first.')
        exit(0)
    return stocks
