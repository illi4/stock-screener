# A class for the simulations
import numpy as np
from itertools import groupby

class Simulation:
    def __init__(self, capital):
        self.current_capital = capital
        self.minimum_value = capital
        self.positions_held = 0
        self.current_positions = set()
        self.capital_values = [capital]
        self.winning_trades_number, self.losing_trades_number = 0, 0
        self.winning_trades, self.losing_trades = [], []
        self.all_trades = []  # to derive further metrics
        self.worst_trade_adjusted, self.best_trade_adjusted = 0, 0
        self.balances = dict()
        self.capital_values.append(self.current_capital)
        (
            self.growth,
            self.win_rate,
            self.max_drawdown,
            self.mom_growth,
            self.max_negative_strike,
        ) = (
            None,
            None,
            None,
            None,
            None,
        )
        # We need to have capital part 'snapshot' as of the time of position entry
        self.capital_per_position = dict()
        # For thresholds
        self.left_of_initial_entries = dict()  # list of initial entries
        self.thresholds_reached = dict()  # for the thresholds reached
        self.entry_prices = dict()  # for the entry prices
        self.entry_dates = dict()  # for the entry dates
        # Another dict for capital values and dates detailed
        self.detailed_capital_values = dict()
        # A dict for failed entry days whatever the condition is
        self.failed_entry_day_stocks = dict()
        # For the failsafe checks
        self.failsafe_stock_trigger = dict()
        self.failsafe_active_dates = dict()  # for dates check
        # For stop losses etc
        self.stop_loss_prices = {}
        self.take_profit_info = {}
        # For executing stop trail updates on the day AFTER the take profit is reached
        self.pending_stop_loss_updates = {}
        self.trailing_stop_active = {}  # for reporting purposes


    def set_take_profit_levels(self, stock, take_profit_variant):
        self.take_profit_info[stock] = {
            'levels': [{'level': level['level'],
                        'exit_proportion': level['exit_proportion'],
                        'reached': False,
                        'price': None,
                        'actual_level': None,
                        'move_stop_price': level.get('move_stop_price', False),
                        'move_stop_from_tp_level': level.get('move_stop_from_tp_level', None)
                        }
                       for level in take_profit_variant['take_profit_values']],
            'taken_profit_proportion': 0
        }

    def check_and_update_take_profit(self, stock, high_price, open_price, take_profit_variant, commission):
        if stock not in self.take_profit_info:
            return False

        entry_price = self.entry_prices[stock]

        # Check each level
        for i, level in enumerate(take_profit_variant['take_profit_values']):
            take_profit_percentage = float(level['level'].strip('%')) / 100
            level_price = entry_price * (1 + take_profit_percentage)

            if high_price >= level_price and not self.take_profit_info[stock]['levels'][i]['reached']:

                price_to_use = max(level_price, open_price)  # if opens higher than the level price, use the open
                exit_proportion = float(level['exit_proportion'].strip('%')) / 100  # this is the proportion of total position
                actual_level = (price_to_use - entry_price) / entry_price

                print(f"-> Taking partial ({exit_proportion:.0%}) profit at {take_profit_percentage:.0%} level @ {price_to_use} ({stock})")
                print(f"-- day open price ${open_price} | level price ${level_price} | using ${price_to_use}")
                print(f"-- actual reflected level reached % is {actual_level:.0%}")

                self.take_profit_info[stock]['levels'][i]['reached'] = True
                self.take_profit_info[stock]['levels'][i]['price'] = price_to_use
                self.take_profit_info[stock]['levels'][i]['actual_level'] = actual_level
                self.take_profit_info[stock]['taken_profit_proportion'] += exit_proportion

                #print(self.take_profit_info[stock])

                # Update stop (change to trailing stop) if required for this level
                if level['move_stop_price']:
                    move_stop_percentage = float(level['move_stop_from_tp_level'].strip('%')) / 100
                    new_stop_level = price_to_use * (1 - move_stop_percentage)
                    self.pending_stop_loss_updates[stock] = new_stop_level
                    print(f"-- scheduled stop level update for {stock} to ${new_stop_level:.2f}")

                if commission > 0:
                    print(f'-- commission ${commission}')
                    self.update_capital(self.current_capital - commission)


    def process_pending_stop_loss_updates(self):
        for stock, new_stop_level in self.pending_stop_loss_updates.items():
            if stock in self.current_positions:
                self.update_stop_level(stock, new_stop_level)
        self.pending_stop_loss_updates.clear()


    def set_stop_loss(self, stock, stop_loss_price):
        self.stop_loss_prices[stock] = stop_loss_price
        print(f"-- stop loss for {stock} set at ${stop_loss_price:.2f}")

    def update_stop_level(self, stock, new_stop_level):
        self.stop_loss_prices[stock] = new_stop_level
        self.trailing_stop_active[stock] = True
        print(f"-- Moved stop level for {stock} to ${new_stop_level:.2f}")

    def update_trade_statistics(self, trade_result_percent, positions_num):
        self.all_trades.append(trade_result_percent)
        if trade_result_percent >= 0:
            self.winning_trades_number += 1
            self.winning_trades.append(trade_result_percent)
            adjusted_profit = trade_result_percent / positions_num
            self.best_trade_adjusted = max(self.best_trade_adjusted, adjusted_profit)
        else:
            self.losing_trades_number += 1
            self.losing_trades.append(trade_result_percent)
            adjusted_profit = trade_result_percent / positions_num
            self.worst_trade_adjusted = min(self.worst_trade_adjusted, adjusted_profit)

    def snapshot_balance(self, current_date_dt):
        self.balances[
            current_date_dt.strftime("%d/%m/%Y")
        ] = self.current_capital  # for the end date
        print("balances:", self.balances)

    def remove_stock_traces(self, stock):
        self.left_of_initial_entries.pop(stock, None)
        self.thresholds_reached.pop(stock, None)
        self.entry_prices.pop(stock, None)
        self.capital_per_position.pop(stock)
        self.thresholds_reached[stock] = []
        self.failed_entry_day_stocks.pop(stock, None)
        self.failsafe_stock_trigger.pop(stock, None)
        self.failsafe_active_dates.pop(stock, None)

    def update_capital(self, new_capital):
        self.current_capital = new_capital
        self.capital_values.append(new_capital)

    def calculate_max_drawdown(self):
        peak = self.capital_values[0]
        max_drawdown = 0
        for value in self.capital_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown

    def calculate_avg_mom_growth(self):
        balances = list(self.balances.values())
        if len(balances) < 2:
            return 0

        total_growth = (balances[-1] - balances[0]) / balances[0]
        num_months = len(balances) - 1
        avg_mom_growth = total_growth / num_months
        return avg_mom_growth

    def calculate_median_mom_growth(self):
        balances = list(self.balances.values())
        if len(balances) < 2:
            return 0

        mom_growths = []
        for i in range(1, len(balances)):
            mom_growth = (balances[i] - balances[i-1]) / balances[i-1]
            mom_growths.append(mom_growth)

        median_mom_growth = np.median(mom_growths)
        return median_mom_growth

    def calculate_longest_negative_strike(self):
        max_negative_strike = 0
        for g, k in groupby(self.all_trades, key=lambda x: x < 0):
            vals = list(k)
            negative_strike_length = len(vals)
            if g and negative_strike_length > max_negative_strike:
                max_negative_strike = negative_strike_length
        return max_negative_strike

    def calculate_metrics(self):
        self.growth = (self.current_capital - self.capital_values[0]) / self.capital_values[0]
        self.win_rate = self.winning_trades_number / (self.winning_trades_number + self.losing_trades_number) if (self.winning_trades_number + self.losing_trades_number) > 0 else 0
        self.max_drawdown = self.calculate_max_drawdown()
        self.mom_growth = self.calculate_median_mom_growth()
        self.average_mom_growth = self.calculate_avg_mom_growth()
        self.max_negative_strike = self.calculate_longest_negative_strike()

    def print_metrics(self):
        print()
        print(f"capital growth/loss: {self.growth:.2%}")
        print(
            f"win rate: {self.win_rate:.2%} | winning_trades: {self.winning_trades_number} | losing trades: {self.losing_trades_number}"
        )
        print(
            f"best trade (adjusted for sizing) {self.best_trade_adjusted:.2%} | worst trade (adjusted for sizing) {self.worst_trade_adjusted:.2%}"
        )
        print(f"max_drawdown: {self.max_drawdown:.2%}")
        print(f"max_negative_strike: {self.max_negative_strike}")
