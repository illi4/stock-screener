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


# Finish
'''
def add_entry(asset_name, entry_price):
    # Use to add an entry for the asset. This is used in pattern day trading limits.
    db.connect()
    entry = Trade.create(
        entry_date_us=current_eastern_date(), asset=asset_name, entry_price=entry_price
    )
    entry.save()
    db.close()


def update_trade(asset_name, exit_price):
    # Use to update an entry for the asset. This is used in pattern day trading limits.
    db.connect()
    last_trade = (
        Trade.select().where(Trade.asset == asset_name).order_by(Trade.id.desc()).get()
    )
    last_trade.exit_date_us = current_eastern_date()
    last_trade.exit_price = exit_price
    last_trade.save()
    db.close()


def number_of_day_trades(start_date):
    # Returns the number of day trades already executed from the start date
    if start_date is None:
        raise Exception("Start date value is missing")
    db.connect()
    day_trades = (
        Trade.select()
        .where(
            (Trade.entry_date_us >= start_date)
            & (Trade.entry_date_us == Trade.exit_date_us)
        )
        .count()
    )
    db.close()
    return day_trades


def get_most_recent_entry_date(asset_name):
    # Returns the most recent entry date for an asset
    db.connect()
    try:
        last_trade = (
            Trade.select()
            .where(
                (Trade.asset == asset_name)
                & Trade.exit_date_us.is_null()  # need the entries only (not closed)
            )
            .order_by(Trade.id.desc())
            .get()
        )
        result = last_trade.entry_date_us
    except peewee.DoesNotExist:
        result = None
    db.close()
    return result


def is_a_day_trade(asset_name):
    # Checks if a trade would be considered a daytrade (exit date and entry dates are the same)
    entry_date = str(get_most_recent_entry_date(asset_name))
    current_date = str(current_eastern_date())
    return entry_date == current_date
'''
