# A class for the simulations

class Simulation:
    def __init__(self, capital):
        self.current_capital = capital
        self.minimum_value = capital
        self.positions_held = 0
        self.current_positions = set()
        self.capital_values = []
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

        # New object
        self.take_profit_info = {}

    # New function, assumes certain structure
    def set_take_profit_levels(self, stock, take_profit_variant):
        self.take_profit_info[stock] = {
            'levels': [{'level': level['level'], 'exit_proportion': level['exit_proportion'], 'reached': False}
                       for level in take_profit_variant['take_profit_values']],
            'total_exit_proportion': 0,
            'total_profit': 0
        }

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
