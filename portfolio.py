import pandas as pd
import yfinance as yf
from datetime import datetime,timedelta
from pathlib import Path

class Portfolio(object):

    def __init__(self, start, end, amount, benchmark='SPY', exposure = 0.25,
                 ftc=0.0, ptc=0.0,price_gap_precent=0.001, verbose=True):
        self.start = start
        self.end = end
        self.initial_amount = amount
        self.amount = amount
        self.cash = amount
        self.exposure = exposure
        self.leverage_amount = 0.02 * self.initial_amount
        self.price_gap_precent = price_gap_precent
        self.ptc = ptc
        self.ftc = ftc
        self.short_trade_money = 0
        self.trade_money_investing = exposure * amount
        self.in_crash = False
        self.exit_crash = False
        self.possible_expousres = [0,0.5,0.7,1,1.2,1.5]
        self.banchmark_prices = []
        self.banchmark_dates = []
        self.money_inside_list = []
        self.net_worth_list = []
        self.exposure_list = []
        self.curr_market_category_list = []
        self.getting_out_rules = []
        self.getting_in_rules = []
        self.getting_out_score_list = []
        self.exposure_range = 0
        self.getting_out_score = 0
        self.getting_in_score = 0
        self.exposure_change_counter = 0
        self.position = 0
        self.trades = 0
        self.total_gain = 0
        self.avg_gain = 0
        self.win_trades = 0
        self.verbose = verbose
        self.stoploss = 0
        self.entry_price = 0
        self.total_days_hold = 0
        self.total_yield = 0
        self.min_exposure = 0
        self.max_exposure = 1.5
        self.benchmark_yield =  0
        self.holdings = {}
        self.short_holdings = {}
        self.today = start
        self.results_path = Path.cwd() / 'Results' 
        self.trade_log = pd.DataFrame(columns=['Date','Buy\Sell','Ticker','Position','Buy\Sell Price','Trade Return','Days Hold',
        'Cash','Net Wealth','Total Trades','Portfolio Yield','Hold Yield','Avg Gain','Avg Days Hold','Win Rate'])
        if not self.results_path.exists():
            self.results_path.mkdir(parents=True)

    def print_balance(self, entry_date):
        print(f'{entry_date} | current cash {self.cash:.2f}')

    def print_net_wealth(self, entry_date,net_wealth):
        print(f'{entry_date} | current net wealth(cash + holdings) {net_wealth:.2f}')

    def total_profit_in_precent(self,today_price):
        self.get_new_wealth(today_price)
        self.total_yield = (self.amount-self.initial_amount)/self.initial_amount
        return self.total_yield

    def balance(self, symbol, entry_date,open_price, exposure_change):
        if exposure_change == 0:
            return
        elif exposure_change>0:
            self.place_buy_order(symbol, entry_date, open_price, exposure_change)
        else:
            self.place_sell_order(symbol, entry_date, open_price, exposure_change)

    def place_buy_order(self, symbol, entry_date,open_price, exposure_change):
         
        entry_price = open_price + (self.price_gap_precent*open_price)
        already_holding = symbol in self.holdings.keys()
        self.exposure += exposure_change
        if already_holding: exist_position = self.holdings[symbol]['Position']
        else: exist_position = 0
        position = int(self.amount*exposure_change/entry_price)
        if position == 0: return
        #NOTE: NEW PART for short

        if symbol in self.short_holdings:
            position_original = position
            position = max(position-self.short_holdings[symbol],0)
            self.short_holdings[symbol] = max(self.short_holdings[symbol]-position_original,0)
            if self.short_holdings[symbol] == 0:
                del self.short_holdings[symbol]

        position_cost = (position * entry_price) + (position * self.ptc) + self.ftc
        if already_holding: current_avg_price = self.holdings[symbol]['Avg Price']
        else: current_avg_price = 0
        if self.exposure > 0: 
            avg_price = (current_avg_price*exist_position+position*entry_price)/(position +exist_position)
            self.holdings[symbol] = {'Avg Price': avg_price, 'Entry Date': entry_date, 'Position':position+exist_position, 'Red Weeks': 0}
        self.cash -= position_cost
            
    # ####################################################################
        self.get_new_wealth(entry_price)
        self.exposure_change_counter += 1
        if self.verbose:
            # new_row = {}
            # new_row['Ticker'] = symbol
            # new_row['Date'] = entry_date
            # new_row['Buy\Sell'] = 'Buy'
            # new_row['Position'] = position
            # new_row['Buy\Sell Price'] = entry_price
            # new_row['Cash'] = self.cash
            # new_row['Net Wealth'] = self.amount
            # self.trade_log = self.trade_log._append(new_row, ignore_index=True)
            # self.trade_log.to_csv(self.results_path / f"exposure_test.csv")
            print('-'*50)
            print(f'{entry_date} | buying {symbol}, {position} units at {entry_price:.2f}')
            self.print_balance(entry_date)
            self.print_net_wealth(entry_date,self.amount)
            print(f'{self.today} | Exposure is {self.trade_money_investing/self.amount:.2f}')

    def sell_original_way(self,symbol, selling_date,open_price, exposure_change):
        selling_price = open_price - (self.price_gap_precent*open_price)
        position = int(self.amount*abs(exposure_change)/selling_price)
        if self.holdings.get(symbol, {}).get('Position', 0) < position or (self.exposure == 0 and self.holdings.get(symbol, {}).get('Position', 0) > 0):
            position = self.holdings[symbol]['Position']
        if position == 0: return
        self.holdings[symbol]['Entry Date'] = selling_date
        self.holdings[symbol]['Position'] -= position
        if self.holdings[symbol]['Position'] == 0:
            self.holdings[symbol]['Avg Price'] = 0
        position_cost = (position * selling_price) - (position * self.ptc) + self.ftc 
        self.cash += position_cost
        self.get_new_wealth(selling_price)
        self.exposure_change_counter += 1

    def sell_short_way(self,symbol,open_price, exposure_change):
        already_holding = symbol in self.short_holdings.keys()


        selling_price = open_price - (self.price_gap_precent*open_price)
        position = int(self.amount*abs(exposure_change)/selling_price)
        if already_holding:
            self.short_holdings[symbol] += position
        else:
            self.short_holdings[symbol] = position
        
        position_cost = (position * selling_price) - (position * self.ptc) + self.ftc 
        self.short_trade_money += position_cost
        self.get_new_wealth(selling_price)
        self.exposure_change_counter += 1

    def place_sell_order(self, symbol, selling_date,open_price, exposure_change):
        exposure_change_cosnt = exposure_change
        if self.exposure>0:
            
            if self.exposure + exposure_change < 0:
                exposure_change = -1*self.exposure
            self.exposure += exposure_change
            self.sell_original_way(symbol, selling_date,open_price, exposure_change)
            if exposure_change_cosnt - exposure_change < 0:
                self.exposure += exposure_change_cosnt - exposure_change
                self.sell_short_way(symbol,open_price, max(exposure_change_cosnt - exposure_change,self.min_exposure))

        elif self.exposure<0:
            self.exposure += exposure_change
            self.sell_short_way(symbol,open_price, max(exposure_change,self.min_exposure))


        # self.sell_short_way(symbol, selling_date,open_price, exposure_change)
        # position = int(self.amount*abs(exposure_change)/selling_price)
        # # if self.holdings[symbol]['Position'] < position or (self.exposure == 0 and self.holdings[symbol]['Position'] > 0):
        # if self.holdings.get(symbol, {}).get('Position', 0) < position or (self.exposure == 0 and self.holdings.get(symbol, {}).get('Position', 0) > 0):
        #     position = self.holdings[symbol]['Position']
        # if position == 0: return
        # self.holdings[symbol]['Entry Date'] = selling_date
        # self.holdings[symbol]['Position'] -= position
        # position_cost = (position * selling_price) - (position * self.ptc) + self.ftc 
        # self.cash += position_cost
        # self.get_new_wealth(selling_price)
        # self.exposure_change_counter += 1
        # if short_action:
        #     self.exposure += tmp
        #     position = int(self.amount*abs(tmp)/selling_price)
        #     self.short_holdings[symbol]['price'] = selling_price
        #     self.short_holdings[symbol]['position'] = position
        # if self.verbose:
            # days_hold = (selling_date - self.holdings[symbol]['Entry Date']).days
            # self.total_days_hold += days_hold
            # self.trades += 1 
            # if(self.holdings[symbol]['Avg Price'] < selling_price):
            #     self.win_trades += 1
            # trade_return = (selling_price - self.holdings[symbol]['Avg Price'])/self.holdings[symbol]['Avg Price']*100
            # self.total_gain += trade_return
            # self.avg_gain = self.total_gain / self.trades
            # self.leverage_amount = 0.02 * self.amount
            # new_row = {}
            # new_row['Trade Return'] = trade_return
            # new_row['Avg Gain'] = self.avg_gain
            # new_row['Days Hold'] = days_hold
            # new_row['Avg Days Hold'] = self.total_days_hold / self.trades
            # new_row['Ticker'] = symbol
            # new_row['Date'] = self.today
            # new_row['Buy\Sell'] = 'Sell'
            # new_row['Position'] = position
            # new_row['Buy\Sell Price'] = selling_price
            # new_row['Cash'] = self.cash
            # new_row['Net Wealth'] = self.amount
            # new_row['Total Trades'] = self.trades
            # new_row['Portfolio Yield'] = (self.amount - self.initial_amount)/self.initial_amount*100
            # try:
            #     new_row['Win Rate'] = self.win_trades/self.trades*100
            # except:    
            #     new_row['Win Rate'] = 0
            # self.trade_log = self.trade_log._append(new_row, ignore_index=True)
            # self.trade_log.to_csv(results_path / f"seq_strategy.csv")
            # print('-'*50)
            # print(f'{self.today} | selling {symbol}, {position} units at {selling_price:.2f}')
            # self.print_balance(self.today)
            # self.print_net_wealth(self.today,self.amount)
            # print(f'{self.today} | Exposure is {self.trade_money_investing/self.amount:.2f}')

    def close_out(self):
        symbol_to_sell =[]
        for symbol in self.holdings.items():
            symbol_to_sell.append(symbol[0])
        for symbol in symbol_to_sell:
            data_daily = yf.download(symbol,start = self.today, end= (self.today +timedelta(days=3)),progress=False)
            self.place_sell_order(symbol,data_daily)
        new_row = {}
        new_row['Hold Yield'] = self.benchmark_yield
        self.trade_log = self.trade_log.append(new_row, ignore_index=True)
        self.trade_log.to_csv(results_path / f"seq_strategy.csv")



    def get_new_wealth(self,today_price):
      
        money_inside = 0
        for symbol in self.holdings.items():
            #NOTE: when use more then one symbol must fit the code and change owned code
            # data_daily = yf.download(symbol[0],start = self.today, end= (self.today +timedelta(days=3)),progress=False)
            # today_price = data_daily['Open'][0]
            position = symbol[1]['Position']
            money_inside += today_price*position
        
        money_inside_short=0
        if self.short_holdings:
            for symbol in self.short_holdings.items():
                #NOTE: when use more then one symbol must fit the code and change owned code
                # data_daily = yf.download(symbol[0],start = self.today, end= (self.today +timedelta(days=3)),progress=False)
                # today_price = data_daily['Open'][0]
                position = symbol[1]
                money_inside_short += today_price*position
 
        self.trade_money_investing = money_inside
        self.short_trade_money = money_inside_short
        self.amount = self.trade_money_investing + self.cash + self.short_trade_money
        if self.trade_money_investing != 0:
            self.exposure = round(self.trade_money_investing/ self.amount,2)
        else:
            self.exposure = round((-1)*self.short_trade_money/ self.amount,2)

    def __str__(self):
            return f"Start: {self.start}, Initial Amount: {self.initial_amount}, " \
                f"Amount: {self.amount}, Cash: {self.cash}, Exposure: {self.exposure}, " \
                f"Leverage Amount: {self.leverage_amount}, Price Gap Percent: {self.price_gap_precent}, " \
                f"PTC: {self.ptc}, FTC: {self.ftc}, Trade Money Investing: {self.trade_money_investing}, "

if __name__ == '__main__':
    results_path = Path.cwd() / 'Results' 
    