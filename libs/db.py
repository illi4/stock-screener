from peewee import *
import peewee
import datetime
import arrow

from libs.read_settings import read_config
config = read_config()

db = SqliteDatabase("stocks.db")


class BaseModel(Model):
    class Meta:
        database = db


class Stock(BaseModel):
    code = CharField()
    name = CharField()
    price = FloatField(null=True)
    volume = FloatField(null=True)
    exchange = CharField()
    type = CharField(null=True)
    date = DateTimeField(default=datetime.datetime.now)


def create_stock_table():
    Stock.create_table()


def delete_all_stocks(exchange):
    query = Stock.delete().where(Stock.exchange == exchange)
    query.execute()


def bulk_add_stocks(stocks_list_of_dict):
    list_length = 100
    # Workaround, see https://github.com/coleifer/peewee/issues/948
    chunks = [
        stocks_list_of_dict[x : x + list_length]
        for x in range(0, len(stocks_list_of_dict), list_length)
    ]
    for chunk in chunks:
        with db.atomic():
            Stock.insert_many(chunk).execute()


def get_stocks(exchange=None, price_min=None, price_max=None, min_volume=None, code=None):
    price_min = 0 if price_min is None else price_min
    price_max = 10e9 if price_max is None else price_max
    min_volume = 0 if min_volume is None else min_volume

    try:

        # If only one stock requested
        if code is not None:
            stocks = Stock.select().where(
                (Stock.code == code)
            )
            return stocks

        if config["filters"]["stocks_only"]:
            print("Excluding funds and ETF per settings")
            stocks = Stock.select().where(
                (Stock.price >= price_min)
                & (Stock.price < price_max)
                & (Stock.volume > min_volume)
                & (Stock.exchange == exchange)
                & (Stock.type == 'Common Stock')
            )
        else:
            stocks = Stock.select().where(
                (Stock.price >= price_min)
                & (Stock.price < price_max)
                & (Stock.volume > min_volume)
                & (Stock.exchange == exchange)
            )
        if len(stocks) == 0:
            print("Warning: no stocks in the database")

    except peewee.OperationalError:
        print("Error: table not found. Update the stocks list first.")
        exit(0)

    return stocks


def get_update_date(exchange):
    try:
        record = Stock.select().where(Stock.exchange == exchange).first()
        record = arrow.get(record.date)
        record = record.replace(tzinfo="Australia/Sydney")
        return record
    except peewee.OperationalError:
        print("Error: table not found. Update the stocks list first.")
        exit(0)
