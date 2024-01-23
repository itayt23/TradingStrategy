import os
from itertools import product
import ta
import send_email as se
from datetime import timedelta,datetime
from pytz import timezone
import pickle
import pandas as pd
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from portfolio import Portfolio
from exceptions import *
from sequencing import SequenceMethodAll
import inspect
import re
import traceback
import numpy as np
from functools import reduce
import time
from multiprocessing import Pool

TIMEZONE = timezone('Israel')
PROGRAM_NAME = (os.path.basename(__file__)).split('.py')[0]
MULTIPLY=1.6
USER_NOTE="VALUES CHECK"


#############################################################################################################################################
############################################################ START RULES MAKER ##############################################################
#############################################################################################################################################

class Rule(object):
    
    def __init__(self,function,buy_sell,weight=1,range_indecator=[]) -> None:
        self.func = function
        self.buy_sell = buy_sell
        self.weight = weight
        self.weight_range = [1,2,3] 
        self.range=[] if range_indecator is None else range_indecator
        if type(self.range) ==list and len(self.range)>0:
            source_code = inspect.getsource(function).replace('"',"'")
            indicator_pattern = re.compile(r"row\['([^']+)'\]")
            indicator = indicator_pattern.findall(source_code)
            if indicator:
                self.range = {indicator[0]:range_indecator}
            else:
                ParameterError('type',"Rule init","can not find indecator")
        self.function_options=None

    def generate_functions(self,just_weight=False,just_range=False):
        if just_weight:
            if self.weight_range == [1]:return [self.replace_params(self.func,{})]
            combinations = product(self.weight_range)
            param_keys = ['weight']
            params = [{key: value for key, value in zip(param_keys, combo)} for combo in combinations]
            result_list = [self.replace_params(self.func,param) for param in params]
            return result_list 
        
        if just_range:
            if self.range == []:return [self.replace_params(self.func,{})]
            combinations = list(product(*self.range.values()))
            params = [{key: value for key, value in zip(self.range.keys(), combo)} for combo in combinations]
            result_list = [self.replace_params(self.func,param) for param in params]
            return result_list
        
        if self.range == [] and self.weight_range == [1]:
            return [self.replace_params(self.func, {})]
        if not self.range:
            combinations = product(self.weight_range)
            param_keys = ['weight']
        else:
            combinations = product(*self.range.values(), self.weight_range)
            param_keys = list(self.range.keys()) + ['weight']

        params = [{key: value for key, value in zip(param_keys, combo)} for combo in combinations]
        result_list = [self.replace_params(self.func, param) for param in params]
        return result_list

    def replace_params(self, func, param_dict):
        weight = param_dict.get('weight', None)
        original_str = inspect.getsource(func)
        pattern = r'\((.*?)\,'
        prefix = "self.add_rule_buy_sell_for_backtest("
        comma_pattern = re.compile(r'\,')
        comma_index = comma_pattern.search(original_str).start()
        suffix = (original_str[comma_index:-1]).strip()
        suffix = suffix.split('#')[0] 
        #exclude ',' from ",type='sell',) " the ,)
        pattern_range = re.compile(r',?\s*range\s*=\s*{[^}]*}')
        suffix = re.sub(pattern_range, '', suffix)
        pattern_range = re.compile(r',?\s*range\s*=\s*\[.*?\]')
        suffix = re.sub(pattern_range, '', suffix)
        suffix = suffix.replace(',)', ')')
        matches = re.findall(pattern, original_str)[0]

        matches = matches.replace(">", " > ")
        matches = matches.replace("<", " < ")
        matches = matches.replace("==", " == ")
        matches = matches.replace("< =", "<=")
        matches = matches.replace("> =", ">=")


        if weight is not None:
            weight_pattern = re.compile(r'\s*weight\s*=\s*(-?\d+(?:\.\d+)?)')
            existing_weight_match = weight_pattern.search(suffix)
            if existing_weight_match:
                suffix = weight_pattern.sub('', suffix)
                suffix = suffix.replace(',)', ')')

        for key, value in param_dict.items():
            if key != 'weight':
                pattern = re.compile(f"row\['{key}'\]\s*(\S+)\s*(-?\d+(?:\.\d+)?)")
                matches = pattern.sub(f"row['{key}'] \\1 {value}", matches)

        full_command = f"{prefix}{matches}"
        if weight is not None:
            full_command += f", weight={weight}"
        full_command += f"{suffix}" 
        full_command = full_command.replace("< =", "<=")
        full_command = full_command.replace("> =", ">=")
        return full_command
    

    #TODO: need to fit this func
    def change_range(self,new_range):
        if type(new_range) ==list:
            source_code = inspect.getsource(self.func).replace('"',"'")
            indicator_pattern = re.compile(r"row\['([^']+)'\]")
            indicator = indicator_pattern.findall(source_code)
            if indicator:
                self.range = {indicator:new_range}
            else:
                ParameterError('type',"Rule init","can not find indecator")
        self.function_options=self.generate_functions()


#############################################################################################################################################
############################################################ END RULES MAKER ################################################################
#############################################################################################################################################

class TradingStrategy(object):

    def __init__(self,benchmark='SPY',threshold_buy=0.4,threshold_sell=0.6,weekdays_to_trade = [0,3],weights_range=[1,2,3],verbose=False) -> None:
        self.rules_to_buy_counter = 0
        self.rules_to_sell_counter = 0
        self.rules_to_buy =[]
        self.rules_to_sell =[]
        self.rules_to_sell_storage=[]
        self.rules_to_buy_storage=[] 
        self.weights_function_to_sell = {}
        self.weights_function_to_buy = {}
        self.threshold_sell = threshold_sell
        self.threshold_buy = threshold_buy
        self.benchmark=benchmark
        self.verbose = verbose
        self.weights_range = weights_range
        self.weekdays_to_trade = weekdays_to_trade 
        # global df_indecator_database
        # df_indecator_database = pd.read_csv('indecator_database.csv')

        self.set_ccrash_conditions()
        self.set_thresholds_exposures()

        
        # self.database = {}
        # for benchmark in benchmarks:
        #     self.database[benchmark] = yf.download(tickers=benchmark, interval='1d',period='max',progress=False)

    def set_weights_range(self,weights_range):
        self.weights_range = weights_range

    def reset_backtest_strategy(self):
        self.rules_to_buy_counter = 0
        self.rules_to_sell_counter = 0
        self.rules_to_buy =[]
        self.rules_to_sell =[]
        self.weights_function_to_sell = {}
        self.weights_function_to_buy = {}

    def build_dataframe(self):
        file_name = 'backtest_daily_' + self.benchmark + '.csv'
        if os.path.exists(file_name):
            self.database = pd.read_csv(file_name)
        else:
            self.database=self.make_df()
        self.database['time'] = pd.to_datetime(self.database['time'], format='%Y-%m-%d')


#############################################################################################################################################
############################################################ START BUILD DF #################################################################
#############################################################################################################################################
        
    def add_cc_dew(self,df:pd.DataFrame):
        cc_df = pd.read_csv('CC.csv')
        dew_df = pd.read_csv('DEW.csv')
        daily_df = df.copy()
        daily_df['time'] = pd.to_datetime(daily_df['time'], format='%m/%d/%Y')
        cc_df['cc dates'] = pd.to_datetime(cc_df['cc dates'], format='%d/%m/%Y')
        dew_df['DEW dates'] = pd.to_datetime(dew_df['DEW dates'], format='%d/%m/%Y')
        cc_merged_df = pd.merge_asof(daily_df.sort_values('time'), cc_df.sort_values('cc dates'),
                                left_on='time', right_on='cc dates', direction='backward')
        cc_merged_df['cc value'] = cc_merged_df['cc value'].ffill()
        
        merged_df = pd.merge_asof(cc_merged_df.sort_values('time'), dew_df.sort_values('DEW dates'),
                                left_on='time', right_on='DEW dates', direction='backward')
        merged_df['DEW'] = merged_df['DEW'].ffill()
        merged_df = merged_df.drop(columns=['cc dates','confirmed_x','confirmed_y','DEW dates',])
        merged_df.rename(columns={'cc value': 'cc', 'DEW': 'dew'}, inplace=True)
        merged_df.dropna(inplace=True)
        return merged_df
        
    def add_rsi(self,df:pd.DataFrame):
        # TODO: check rsi diff
        # TODO: check rsi sma
        df['rsi_daily'] = round(ta.momentum.RSIIndicator(df['Close'], window=14,fillna=True).rsi(),2)
        df['rsi_daily_sma'] = df['rsi_daily'].rolling(window=5).mean().round(2)

        df_week = yf.download(tickers=self.benchmark, interval='1wk',period='max',progress=False,rounding=True)
        df_week['rsi_weekly'] = round(ta.momentum.RSIIndicator(df_week['Close'], window=14,fillna=True).rsi(),2)
        df_week.reset_index(inplace=True)
        df_week['Date_w'] = pd.to_datetime(df_week['Date'])
        df_week = df_week[['Date_w','rsi_weekly']].copy()
        df_week['rsi_weekly_shifted'] = df_week['rsi_weekly'].shift(1)
        df_week['rsi_weekly_sma'] = df_week['rsi_weekly_shifted'].rolling(window=5).mean().round(2)
        df_month = yf.download(tickers=self.benchmark, interval='1mo',period='max',progress=False)
        df_month['rsi_monthly'] = ta.momentum.RSIIndicator(df_month['Close'], window=14).rsi()
        df_month.reset_index(inplace=True)
        df_month['Date_m'] = pd.to_datetime(df_month['Date'])
        df_month = df_month[['Date_m','rsi_monthly']].copy()
        df_month['rsi_monthly_shifted'] = df_month['rsi_monthly'].shift(1)
        df_month['rsi_monthly_sma'] = df_month['rsi_monthly_shifted'].rolling(window=5).mean().round(2)
        df_with_rsi = pd.merge_asof(df.sort_values('time'), df_week.sort_values('Date_w'),
                                    left_on='time', right_on='Date_w',direction='backward')
        df_with_rsi = pd.merge_asof(df_with_rsi.sort_values('time'), df_month.sort_values('Date_m'),
                                    left_on='time', right_on='Date_m',direction='backward')
        df_with_rsi.drop(columns=['Date_w','Date_m','rsi_monthly','rsi_weekly'],inplace=True)
        #rename columns
        df_with_rsi.rename(columns={'rsi_weekly_shifted': 'rsi_weekly', 'rsi_monthly_shifted': 'rsi_monthly'}, inplace=True)
        return df_with_rsi
        
    def add_trend_to_god(self,daily_df:pd.DataFrame):
        sequencing = SequenceMethodAll(self.benchmark)
        trend_to_god_day = sequencing.sequence_daily[['Date','Sequence']].copy()
        trend_to_god_day.rename(columns={'Sequence': 'trend_to_god_day','Date':'Date_day'}, inplace=True)
        trend_to_god_day['Date_day'] = pd.to_datetime(trend_to_god_day['Date_day'])
        trend_to_god_week = sequencing.sequence_weekly[['Date','Sequence']].copy()
        trend_to_god_week.rename(columns={'Sequence': 'trend_to_god_week','Date':'Date_week'}, inplace=True)
        trend_to_god_week['Date_week'] = pd.to_datetime(trend_to_god_week['Date_week'])
        trend_to_god_month = sequencing.sequence_monthly[['Date','Sequence']].copy()
        trend_to_god_month.rename(columns={'Sequence': 'trend_to_god_month','Date':'Date_month'}, inplace=True)
        trend_to_god_month['Date_month'] = pd.to_datetime(trend_to_god_month['Date_month'])
        df_with_trend = pd.merge_asof(daily_df.sort_values('time'), trend_to_god_day.sort_values('Date_day'),
                                    left_on='time', right_on='Date_day',direction='backward')
        df_with_trend['trend_to_god_day'] = df_with_trend['trend_to_god_day'].shift(1)
        df_with_trend = pd.merge_asof(df_with_trend.sort_values('time'), trend_to_god_week.sort_values('Date_week'),
                                    left_on='time', right_on='Date_week',direction='backward')
        df_with_trend = pd.merge_asof(df_with_trend.sort_values('time'), trend_to_god_month.sort_values('Date_month'),
                                    left_on='time', right_on='Date_month',direction='backward')
        df_with_trend = df_with_trend[['Open', 'Close', 'time', 'cc', 'dew', 'trend_to_god_day',
        'trend_to_god_week', 'trend_to_god_month']]
        df_with_trend.dropna(inplace=True)
        return df_with_trend

    def add_bb(self,df:pd.DataFrame):
        df['BB_Middle_Band'] = round(df['Close'].rolling(window=20).mean(),2)
        df['BB_Upper_Band'] = round(df['BB_Middle_Band'] + 2 * df['Close'].rolling(window=20).std(),2)
        df['BB_Lower_Band'] = round(df['BB_Middle_Band'] - 2 * df['Close'].rolling(window=20).std(),2)

        #add column that 1 if close was below lower band in the last 10 candles and 0 otherwise
        df['daily_below_lower_band'] = df.apply(lambda row: 1 if any(row['Close'] < row['BB_Lower_Band'] for _, row in df.iloc[max(0, row.name-10):row.name].iterrows()) else 0, axis=1)
        df['daily_above_upper_band'] = df.apply(lambda row: 1 if any(row['Close'] > row['BB_Upper_Band'] for _, row in df.iloc[max(0, row.name-10):row.name].iterrows()) else 0, axis=1)
        
        #add weekly bb
        df_week = yf.download(tickers=self.benchmark, interval='1wk',period='max',progress=False,rounding=True)
        df_week['weekly_BB_Middle_Band'] = round(df_week['Close'].rolling(window=20).mean(),2)
        df_week['weekly_BB_Upper_Band'] = round(df_week['weekly_BB_Middle_Band'] + 2 * df_week['Close'].rolling(window=20).std(),2)
        df_week['weekly_BB_Lower_Band'] = round(df_week['weekly_BB_Middle_Band'] - 2 * df_week['Close'].rolling(window=20).std(),2)
        df_week.drop(columns=['weekly_BB_Middle_Band'],inplace=True)
        df_week.reset_index(inplace=True)
        df_week['Date_w'] = pd.to_datetime(df_week['Date'])
        df_week = df_week[['Date_w','weekly_BB_Upper_Band','weekly_BB_Lower_Band']].copy()
        df_week['weekly_BB_Upper_Band_shifted'] = df_week['weekly_BB_Upper_Band'].shift(1)
        df_week['weekly_BB_Lower_Band_shifted'] = df_week['weekly_BB_Lower_Band'].shift(1)
        df_week = df_week[['Date_w','weekly_BB_Upper_Band_shifted','weekly_BB_Lower_Band_shifted']].copy()
        df = pd.merge_asof(df.sort_values('time'), df_week.sort_values('Date_w'),
                                        left_on='time', right_on='Date_w',direction='backward')
        df.rename(columns={'weekly_BB_Upper_Band_shifted': 'weekly_BB_Upper_Band','weekly_BB_Lower_Band_shifted': 'weekly_BB_Lower_Band'}, inplace=True)
        df.drop(columns=['Date_w'],inplace=True)

        df['weekly_below_lower_band'] = df.apply(lambda row: 1 if any(row['Close'] < row['weekly_BB_Lower_Band'] for _, row in df.iloc[max(0, row.name-4):row.name].iterrows()) else 0, axis=1)
        df['weekly_above_upper_band'] = df.apply(lambda row: 1 if any(row['Close'] > row['weekly_BB_Upper_Band'] for _, row in df.iloc[max(0, row.name-4):row.name].iterrows()) else 0, axis=1)
        #add monthly bb
        df_month = yf.download(tickers=self.benchmark, interval='1mo',period='max',progress=False,rounding=True)
        df_month['monthly_BB_Middle_Band'] = round(df_month['Close'].rolling(window=20).mean(),2)
        df_month['monthly_BB_Upper_Band'] = round(df_month['monthly_BB_Middle_Band'] + 2 * df_month['Close'].rolling(window=20).std(),2)
        df_month['monthly_BB_Lower_Band'] = round(df_month['monthly_BB_Middle_Band'] - 2 * df_month['Close'].rolling(window=20).std(),2)
        df_month.drop(columns=['monthly_BB_Middle_Band'],inplace=True)
        df_month.reset_index(inplace=True)
        df_month['Date_m'] = pd.to_datetime(df_month['Date'])
        df_month = df_month[['Date_m','monthly_BB_Upper_Band','monthly_BB_Lower_Band']].copy()
        df_month['monthly_BB_Upper_Band_shifted'] = df_month['monthly_BB_Upper_Band'].shift(1)
        df_month['monthly_BB_Lower_Band_shifted'] = df_month['monthly_BB_Lower_Band'].shift(1)
        df_month = df_month[['Date_m','monthly_BB_Upper_Band_shifted','monthly_BB_Lower_Band_shifted']].copy()
        df = pd.merge_asof(df.sort_values('time'), df_month.sort_values('Date_m'),
                                        left_on='time', right_on='Date_m',direction='backward')
        df.rename(columns={'monthly_BB_Upper_Band_shifted': 'monthly_BB_Upper_Band','monthly_BB_Lower_Band_shifted': 'monthly_BB_Lower_Band'}, inplace=True)
        df.drop(columns=['Date_m'],inplace=True)

        df['monthly_below_lower_band'] = df.apply(lambda row: 1 if any(row['Close'] < row['monthly_BB_Lower_Band'] for _, row in df.iloc[max(0, row.name-2):row.name].iterrows()) else 0, axis=1)
        df['monthly_above_upper_band'] = df.apply(lambda row: 1 if any(row['Close'] > row['monthly_BB_Upper_Band'] for _, row in df.iloc[max(0, row.name-2):row.name].iterrows()) else 0, axis=1)
        
        return df
    
    def add_macd(self,df:pd.DataFrame):
        df['12ema'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['26ema'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = round(df['12ema'] - df['26ema'],2)
        df['signal'] = round(df['macd'].ewm(span=9, adjust=False).mean(),2)
        df['daily_macd_diff'] = round(df['macd'] - df['signal'],2)
        df.drop(columns=['12ema','26ema','macd','signal'],inplace=True)

        df_week = yf.download(tickers=self.benchmark, interval='1wk',period='max',progress=False,rounding=True)
        df_week['12ema'] = df_week['Close'].ewm(span=12, adjust=False).mean()
        df_week['26ema'] = df_week['Close'].ewm(span=26, adjust=False).mean()
        df_week['macd'] = round(df_week['12ema'] - df_week['26ema'],2)
        df_week['signal'] = round(df_week['macd'].ewm(span=9, adjust=False).mean(),2)
        df_week['weekly_macd_diff'] = round(df_week['macd'] - df_week['signal'],2)
        df_week.drop(columns=['12ema','26ema','macd','signal'],inplace=True)
        df_week.reset_index(inplace=True)
        df_week['Date_w'] = pd.to_datetime(df_week['Date'])
        df_week = df_week[['Date_w','weekly_macd_diff']].copy()
        df_week['weekly_macd_diff_shifted'] = df_week['weekly_macd_diff'].shift(1)
        df_week = df_week[['Date_w','weekly_macd_diff_shifted']].copy()

        df = pd.merge_asof(df.sort_values('time'), df_week.sort_values('Date_w'),
                                    left_on='time', right_on='Date_w',direction='backward')
        
        df_month = yf.download(tickers=self.benchmark, interval='1mo',period='max',progress=False,rounding=True)
        df_month['12ema'] = df_month['Close'].ewm(span=12, adjust=False).mean()
        df_month['26ema'] = df_month['Close'].ewm(span=26, adjust=False).mean()
        df_month['macd'] = round(df_month['12ema'] - df_month['26ema'],2)
        df_month['signal'] = round(df_month['macd'].ewm(span=9, adjust=False).mean(),2)
        df_month['monthly_macd_diff'] = round(df_month['macd'] - df_month['signal'],2)
        df_month.drop(columns=['12ema','26ema','macd','signal'],inplace=True)
        df_month.reset_index(inplace=True)
        df_month['Date_m'] = pd.to_datetime(df_month['Date'])
        df_month = df_month[['Date_m','monthly_macd_diff']].copy()
        df_month['monthly_macd_diff_shifted'] = df_month['monthly_macd_diff'].shift(1)
        df_month = df_month[['Date_m','monthly_macd_diff_shifted']].copy()

        df = pd.merge_asof(df.sort_values('time'), df_month.sort_values('Date_m'),
                                        left_on='time', right_on='Date_m',direction='backward') 
        
        df.drop(columns=['Date_w','Date_m'],inplace=True)
        df.rename(columns={'weekly_macd_diff_shifted': 'weekly_macd_diff', 'monthly_macd_diff_shifted': 'monthly_macd_diff'}, inplace=True)

        return df
    
    def add_vix(self,df:pd.DataFrame):
        df_vix = yf.download(tickers='^VIX', interval='1d',period='max',progress=False)
        df_vix.reset_index(inplace=True)
        df_vix['time'] = pd.to_datetime(df_vix['Date'])
        df_vix = df_vix[['time', 'Open', 'High', 'Low', 'Close']]
        df = df.merge(df_vix[['time', 'Close']], on='time', suffixes=('', '_vix'), how='left')
        df.rename(columns={'Close_vix': 'vix'}, inplace=True)
        df['vix_change_5'] = df['vix'].pct_change(periods=5).round(2)*100
        df['vix_change_3'] = df['vix'].pct_change(periods=3).round(2)*100

        return df

    def add_last_candles_change(self,df:pd.DataFrame,period_month=15,period_year=252):
        df['last_month_change'] = df['Close'].pct_change(periods=period_month).round(2)*100
        df['last_year_change'] = df['Close'].pct_change(periods=period_year).round(2)*100
        return df

    def add_highes_vix(self,df:pd.DataFrame):
        df['52w_high_vix'] = df['VIX_SMA5'].rolling(window=252).max()
        df['26w_high_vix'] = df['VIX_SMA5'].rolling(window=126).max()
        df['13w_high_vix'] = df['VIX_SMA5'].rolling(window=63).max()
        return df
    
    def add_sma(self,df:pd.DataFrame):
        df['SMA5'] = df['Close'].rolling(window=5).mean().round(2)
        df['SMA10'] = df['Close'].rolling(window=10).mean().round(2)
        df['SMA20'] = df['Close'].rolling(window=20).mean().round(2)
        df['SMA50'] = df['Close'].rolling(window=50).mean().round(2)
        df['VIX_SMA5'] = df['vix'].rolling(window=5).mean().round(2)
        df['VIX_SMA10'] = df['vix'].rolling(window=10).mean().round(2)
        return df
        
    def add_tense(self,daily_df:pd.DataFrame):
        df_month = yf.download(tickers=self.benchmark, interval='1mo',period='max',progress=False,rounding=True)
        df_month['close_month_before'] = df_month['Close'].shift(1)
        df_month['change'] = (df_month['Close'] - df_month['close_month_before'])/df_month['close_month_before']*100
        df_month['consecutive_green'] = df_month.apply(lambda row: 1 if row['change'] > 1 else 0, axis=1)
        # Reset cumulative sum when encountering 0
        df_month['consecutive_green_reset'] = df_month['consecutive_green'].eq(0).cumsum()
        df_month['consecutive_green_sum'] = df_month.groupby('consecutive_green_reset')['consecutive_green'].cumsum()
        df_month['consecutive_green_sum'] = df_month['consecutive_green_sum'].shift(1)
        df_month.drop(columns=['consecutive_green_reset','consecutive_green'],inplace=True)
        df_month.rename(columns={'consecutive_green_sum': 'consecutive_green'}, inplace=True)
        df_month.reset_index(inplace=True)
        df_month['Date_m'] = pd.to_datetime(df_month['Date'])
        df_month = df_month[['Date_m','consecutive_green']].copy()
        daily_df = pd.merge_asof(daily_df.sort_values('time'), df_month.sort_values('Date_m'),
                                    left_on='time', right_on='Date_m',direction='backward')
        daily_df.drop(columns=['Date_m'],inplace=True)

        return daily_df

    def convert_to_binary(self,df_origin:pd.DataFrame):
        df = df_origin.copy()
        df = df[(df['trend_to_god_week'] != 0) | (df['trend_to_god_day'] != 0)]
        df['trend_to_god_month'] =df['trend_to_god_month'].apply(lambda x: 1 if x>0 else 0)
        df['trend_to_god_week'] =df['trend_to_god_week'].apply(lambda x: 1 if x>0 else 0)
        df['trend_to_god_day'] =df['trend_to_god_day'].apply(lambda x: 1 if x>0 else 0)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        df['close_yesterday'] = df['close'].shift(1)
        df['change'] = (df['close'] - df['close_yesterday'])/df['close_yesterday']*100
        df = df.reset_index(drop=True) 

        return df
    
    def shift_df(self,df:pd.DataFrame):
        df_shifted = df.copy()
        columns_need_to_shift = ['cc','dew','ccrash','bb_middle_band','bb_upper_band','bb_lower_band','daily_macd_diff','vix','vix_change_5'
                                ,'vix_change_3','vix_sma5','vix_sma10','sma5','sma10','sma20','sma50','52w_high_vix','26w_high_vix','13w_high_vix']
        df_shifted[columns_need_to_shift] = df_shifted[columns_need_to_shift].shift(1)
        df_shifted = df_shifted.reset_index(drop=True)
        result_df = pd.concat([df[['time', 'open']], df_shifted.drop(['time', 'open'], axis=1)], axis=1)
        result_df = result_df[:-1]
        return result_df
    
    def make_df(self)-> pd.DataFrame: 
        backtest_df = yf.download(tickers=self.benchmark, interval='1d',period='max',progress=False)
        backtest_df.reset_index(inplace=True)
        backtest_df['Date'] = pd.to_datetime(backtest_df['Date'])
        backtest_df['time'] = backtest_df['Date'].dt.strftime('%m/%d/%Y')
        backtest_df = backtest_df[['time','Open','Close']]

        backtest_df = self.add_cc_dew(backtest_df)
        backtest_df = self.add_trend_to_god(backtest_df)
        backtest_df = self.add_rsi(backtest_df)
        backtest_df = self.add_bb(backtest_df)
        backtest_df = self.add_macd(backtest_df)
        backtest_df = self.add_vix(backtest_df)
        backtest_df = self.add_sma(backtest_df)
        backtest_df = self.add_last_candles_change(backtest_df)
        backtest_df = self.add_highes_vix(backtest_df)
        backtest_df = self.add_tense(backtest_df)

        backtest_df.columns = backtest_df.columns.str.lower()
        backtest_df = self.convert_to_binary(backtest_df)

        backtest_df['ccrash'] = backtest_df.apply(lambda row :0 if all(ccrash_condition(row) for ccrash_condition in self.ccrash_conditions) else 1,axis=1)
        backtest_df = self.shift_df(backtest_df)
        backtest_df.dropna(inplace=True)
        backtest_df.reset_index(drop=True, inplace=True)

        name_csv = 'backtest_daily_' + self.benchmark + '.csv'
        backtest_df.to_csv(name_csv)
        return backtest_df

#############################################################################################################################################
############################################################ END BUILD DF ###################################################################
#############################################################################################################################################

#############################################################################################################################################
############################################################ START VISUALIZATION DF #########################################################
#############################################################################################################################################
   
    def save_portfolio_plot_interactive(self,portfolio:Portfolio, filename='portfolio_plot.html'):
        normalized_benchmark_prices = [price / portfolio.banchmark_prices[0] for price in portfolio.banchmark_prices]
        normalized_net_worth = [worth / portfolio.net_worth_list[0] for worth in portfolio.net_worth_list]
        yearly_yields_df = self.cal_yield_each_year(portfolio)
        fig = make_subplots(rows=3, cols=1, subplot_titles=['Normalized Benchmark Prices and Net Worth', 'Exposure'])

        fig.add_trace(go.Scatter(x=portfolio.banchmark_dates, y=normalized_benchmark_prices, mode='lines', name='Normalized Benchmark Prices'), row=1, col=1)
        fig.add_trace(go.Scatter(x=portfolio.banchmark_dates, y=normalized_net_worth, mode='lines', name='Normalized Net Worth', line=dict(dash='dash')), row=1, col=1)


        fig.add_trace(go.Scatter(
            x=portfolio.banchmark_dates,
            y=portfolio.exposure_list,
            mode='lines',
            name='Exposure',
            hovertemplate='Date: %{x}<br>Exposure: %{y}<br>Getting Out Score: %{customdata}',
            customdata=portfolio.getting_out_score_list,
            legendgroup='group1',  # Assign the same legendgroup to group traces
        ), row=2, col=1)

        fig.add_annotation(
            go.layout.Annotation(
                x=0.5,
                y=1.12,
                xref="paper",
                yref="paper",
                text=f"Benchmark Yield: {round(portfolio.benchmark_yield * 100,2)}%",
                showarrow=False,
            )
        )
        fig.add_annotation(
            go.layout.Annotation(
                x=0.5,
                y=1.10,
                xref="paper",
                yref="paper",
                text=f"Benchmark Yield with 150% Exposure: {round(portfolio.benchmark_yield * 100 * 1.5,2)}%",
                showarrow=False,
            )
        )
        fig.add_annotation(
            go.layout.Annotation(
                x=0.5,
                y=1.06,
                xref="paper",
                yref="paper",
                text=f"Our Exposure Model Yield: {round(portfolio.total_yield * 100,2)}%",
                showarrow=False,
            )
        )

        table_trace = go.Table(
            header=dict(values=['Year', 'Benchmark Yield', 'Your Yield','Yield Diff']),
            cells=dict(values=[yearly_yields_df.index,
                            yearly_yields_df['benchmark_yield'],
                            yearly_yields_df['yield'],
                            yearly_yields_df['yield_diff']]),
            domain=dict(x=[0, 1], y=[0, 0.3])  # Adjust the y values as needed, set y to 0 to place it at the bottom
        )
    
        fig.add_trace(table_trace)

        fig.update_layout(title_text='Interactive Portfolio Plot', showlegend=True)
        fig.write_html(filename)

    def cal_yield_each_year(self,portfolio:Portfolio):
        def find_nearest_date(date, date_list):
            return min(date_list, key=lambda x: abs(x - date))
        
        years = [date.year for date in portfolio.banchmark_dates]
        years = list(set(years))
        years.sort()
        years_yield = {}
        for year in years:
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)

            if start_date in portfolio.banchmark_dates:
                start_index = portfolio.banchmark_dates.index(start_date)
            else:
                nearest_start_date = find_nearest_date(start_date, portfolio.banchmark_dates)
                start_index = portfolio.banchmark_dates.index(nearest_start_date)

            if end_date in portfolio.banchmark_dates:
                end_index = portfolio.banchmark_dates.index(end_date)
            else:
                nearest_end_date = find_nearest_date(end_date, portfolio.banchmark_dates)
                end_index = portfolio.banchmark_dates.index(nearest_end_date)

            years_yield[year] = (portfolio.net_worth_list[end_index] / portfolio.net_worth_list[start_index] - 1)*100

        years_yield_benchmark = {}
        for year in years:
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)

            if start_date in portfolio.banchmark_dates:
                start_index = portfolio.banchmark_dates.index(start_date)
            else:
                nearest_start_date = find_nearest_date(start_date, portfolio.banchmark_dates)
                start_index = portfolio.banchmark_dates.index(nearest_start_date)

            if end_date in portfolio.banchmark_dates:
                end_index = portfolio.banchmark_dates.index(end_date)
            else:
                nearest_end_date = find_nearest_date(end_date, portfolio.banchmark_dates)
                end_index = portfolio.banchmark_dates.index(nearest_end_date)

            years_yield_benchmark[year] = (portfolio.banchmark_prices[end_index] / portfolio.banchmark_prices[start_index] - 1)*100

        # save to csv
        df = pd.DataFrame.from_dict(years_yield, orient='index', columns=['yield'])
        df['benchmark_yield'] = years_yield_benchmark.values()
        df['yield_diff'] = df['yield'] - df['benchmark_yield']
        return df

    
#############################################################################################################################################
############################################################ END VISUALIZATION DF ###########################################################
#############################################################################################################################################
    

#############################################################################################################################################
############################################################ START BACKTEST DF ##############################################################
#############################################################################################################################################
    def update_portfolio(self,row):
        # global getting_out_rules
        if row.name == 0:
            self.portfolio.place_buy_order(self.benchmark, self.portfolio.start, row['open'], 1)
        else:
            dayweek = row['time'].weekday()
            self.portfolio.today = row['time']
            self.portfolio.get_new_wealth(row['open'])

            if dayweek in self.weekdays_to_trade:
                current_exporsure = self.portfolio.exposure
                exposure_change = 0
                exposure_wanted = -1            
                self.portfolio.getting_out_score = self.get_getting_out_score(row)
                if not self.portfolio.in_crash:
                    exposure_wanted = self.get_exposure_wanted()
                else:
                    self.portfolio.getting_in_score = self.get_getting_in_score(row)
                    if self.portfolio.getting_in_score >= self.threshold_buy:
                        exposure_wanted = self.get_exposure_wanted()
                        self.portfolio.in_crash = False
                if exposure_wanted != -1:
                    exposure_change = round(exposure_wanted - current_exporsure,2)
                self.portfolio.balance(self.benchmark, row['time'], row['open'], exposure_change)

        if self.verbose:
            self.portfolio.getting_out_score_list.append(self.portfolio.getting_out_score)
            self.portfolio.banchmark_prices.append(row['open'])
            self.portfolio.banchmark_dates.append(row['time'])
            self.portfolio.money_inside_list.append(self.portfolio.trade_money_investing)
            self.portfolio.net_worth_list.append(self.portfolio.amount)
            self.portfolio.exposure_list.append(self.portfolio.exposure)

    def set_ccrash_conditions(self,lambda_functions = (lambda row:row['sma5']<row['sma10'] and row['vix_sma5'] > row['vix_sma10'] and row['vix']>22)):
        if type(lambda_functions) != list:
            self.ccrash_conditions = [lambda_functions]
        else:
            self.ccrash_conditions = lambda_functions
    
    def get_exposure_wanted(self):
        
        for threshold, exposure in zip(self.thresholds, self.target_exposures):
            if self.portfolio.getting_out_score <= threshold:
                self.portfolio.exposure_range = exposure
                return exposure
        else:
            self.portfolio.exposure_range = 0
            self.portfolio.in_crash = True
            return 0

    def is_function_valid(self,lambda_function):
        #TODO: finish the function - check logic relations
        source_code = inspect.getsource(lambda_function)
        self.get_relations(source_code)
        involved_indecators = self.get_involved_indecators(source_code)
        answers =[]
        return True
        
    def is_valid_number(value, min_value, max_value, continuous_discrete):
        if continuous_discrete == 'continuous':
            return min_value <= value <= max_value
        elif continuous_discrete == 'discrete':
            return value in range(min_value, max_value + 1)
        else:
            return False
        
    def get_involved_indecators(self,source_code):
        source_code = source_code.replace('"',"'")
        indicator_pattern = re.compile(r"row\['([^']+)'\]")
        indicators = indicator_pattern.findall(source_code)
        return list(set(indicators))

    def add_rule_buy_sell(self,function_represet_rule,type="both",weight=1,range=[]):
        if type not in ['both','buy','sell']:
            raise ParameterError('type',"add_rule_buy_sell","Must be one of these both, buy, sell")
        if weight <= 0:
            raise ParameterError('weight',"add_rule_buy_sell","Must be positive number not zero")
        # if not self.is_function_valid(function_represet_rule):
        #     raise FunctionNotValid(lambda_function=inspect.getsource(function_represet_rule),function='add_rule_buy_sell')
        rule = Rule(function_represet_rule,type,weight,range)
        if type =='sell' or type =='both':
            self.rules_to_sell_storage.append(rule)
        if type =='buy' or type =='both':
            self.rules_to_buy_storage.append(rule)
        self.add_rule_buy_sell_for_backtest(rule.func,type,weight)

    def add_rule_buy_sell_for_backtest(self,rule,type,weight=1):
        if type =='sell' or type =='both':
            self.weights_function_to_sell[rule] = weight
            self.rules_to_sell.append(rule)
            self.rules_to_sell_counter +=weight
        if type =='buy' or type =='both':
            self.weights_function_to_buy[rule] = weight
            self.rules_to_buy.append(rule)
            self.rules_to_buy_counter +=weight


    def get_getting_out_score(self, row):
        scores = np.array([rule(row) for rule in self.rules_to_sell])
        weighted_scores = scores * np.array([self.weights_function_to_sell[rule] for rule in self.rules_to_sell])
        score = round(np.sum(weighted_scores) / self.rules_to_sell_counter, 2)
        return score

    def get_getting_in_score(self,row):
        score = sum(rule(row)*self.weights_function_to_buy[rule] for rule in self.rules_to_buy)
        return  round(score/self.rules_to_buy_counter,2)

    def set_thresholds_exposures(self,thresholds =[0.4,0.5,0.6,0.7,0.8], target_exposures=[1.5,1.2,1,0.7,0.5]):
        #NOTE: can be thinking as function f(x)=1-x
        if len(thresholds) != len(target_exposures):
            raise ParameterError('thresholds target_exposures',"set_thresholds_exposures","Must be same length")
        self.thresholds = sorted(thresholds)
        self.target_exposures = sorted(target_exposures,reverse=True)

    def run_backtest(self,start_date='2010-01-01',verbose=False):
        self.verbose = verbose
        self.database = self.database[self.database['time'] >= start_date]
        self.database.reset_index(drop=True, inplace=True)
        start_date =self.database.at[0,'time']
        end_date = self.database.at[len(self.database)-1,'time']
        benchmark_start = pd.DataFrame(yf.download(tickers=benchmark, start=start_date, end= start_date + timedelta(days=3),interval='1d',progress=False)).dropna()
        benchmark_end = pd.DataFrame(yf.download(tickers=benchmark, start=end_date +timedelta(days=-1), end= end_date,interval='1d',progress=False)).dropna()
        benchmark_yield = (benchmark_end.iloc[0]['Open'] -benchmark_start.iloc[0]['Open'])/ benchmark_start.iloc[0]['Open']

        self.start_backtest_strategy(start_date,end_date,benchmark_yield)
        

        
    def start_backtest_strategy(self,start_date = '2010-01-01',end_date = '2023-12-01',benchmark_yield=None):
        self.portfolio = Portfolio(start=start_date,end=end_date,exposure=0,benchmark=self.benchmark,amount=1000000,ptc=0.005,verbose=False)
        self.portfolio.benchmark_yield = benchmark_yield
        self.database.apply(lambda row : self.update_portfolio(row), axis=1) 
        change = self.portfolio.total_profit_in_precent(self.database.at[len(self.database)-1,'close'])

        if self.verbose:
            self.save_portfolio_plot_interactive(self.portfolio,filename='Results/backtest_' +start_date.strftime("%Y-%m-%d")+'_'+ self.benchmark + '.html')
            print(f'Backtest for {self.benchmark} from {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')
            print('benchmark_yield: {:.2f}%'.format(self.portfolio.benchmark_yield*100))
            print('benchmark_yield with 150% exposure hold: {:.2f}%' .format(self.portfolio.benchmark_yield*100*1.5))
            print('Our Exposure Model Yield: {:.2f}%'.format(change*100))
            print('-'*50)
        
        
        return change
    def cartesian_product_generator(self,list_of_lists):
        if not list_of_lists:
            yield []
        else:
            first_list, *rest_lists = list_of_lists
            for element in first_list:
                for rest_combination in self.cartesian_product_generator(rest_lists):
                    yield [element] + rest_combination

    def combination_generator(self, list_of_lists):
        if not list_of_lists:
            yield []
        else:
            for combination in product(*list_of_lists):
                yield list(combination)

    def estimated_runtime(self,benchmarks,start_date='2010-01-01',end_date = '2023-12-01'):
        each_iter = self.cal_each_iter_time(start_date,end_date)
        sell_combinations = [len(combi.function_options) for combi in  self.rules_to_sell_storage]
        buy_combinations = [len(combi.function_options) for combi in  self.rules_to_buy_storage]
        result = reduce(lambda x, y: x * y, sell_combinations)
        result *= reduce(lambda x, y: x * y, buy_combinations)
        result *= len(benchmarks)
        time_to_run=result*each_iter
        if time_to_run >3600:
            # se.send_me_email(f"Estimated time finish in {round(result*each_iter/3600,2)} Hours - {PROGRAM_NAME} {USER_NOTE}")
            print(round(result*each_iter/3600,2)," Hours")
        else:
            # se.send_me_email(f"Estimated time finish in {round(result*each_iter/60,2)} Minutes - {PROGRAM_NAME} {USER_NOTE}")
            print(round(result*each_iter/60,2)," Minutes")


    def cal_each_iter_time(self,start_date='2010-01-01',end_date = '2023-12-01'):
        len_range=25
        start_time = time.time()
        for i in range(len_range):
            self.start_backtest_strategy(start_date,end_date)
        end_time = time.time()
        each_iter = (end_time - start_time)/len_range
        # print(f"Estimated time for each iteration({len_range} iteration): ",each_iter) 
        # se.send_me_email(f"Estimated time for each iteration({len_range} iteration): ",each_iter)
        return each_iter


    def optimizer(self,start_date='2010-01-01',optimize_weight=False,optimize_val=False,estimated_runtime=False):
        good_combinations = [] 
        generator_list = []
        file_path = f"Results/{self.benchmark}_best_combination_from_{start_date}.pkl"
        iteration_count = 0
        save_interval = 1000000
        if not os.path.exists("Results"): os.makedirs("Results")

        generator_rules_combi = self.rules_to_sell_storage + self.rules_to_buy_storage
        for rule in generator_rules_combi:
            if optimize_weight: rule.function_options = rule.generate_functions(just_weight=True); generator_list.append(rule.function_options)
            elif optimize_val: rule.function_options = rule.generate_functions(just_range=True); generator_list.append(rule.function_options)
            else: rule.function_options = rule.generate_functions(); generator_list.append(rule.function_options)

        self.database = self.database[self.database['time'] >= start_date]
        self.database.reset_index(drop=True, inplace=True)
        start_date =self.database.at[0,'time']
        end_date = self.database.at[len(self.database)-1,'time']

        if estimated_runtime: self.estimated_runtime(benchmarks,start_date,end_date)

        benchmark_start = pd.DataFrame(yf.download(tickers=benchmark, start=start_date, end= start_date + timedelta(days=3),interval='1d',progress=False)).dropna()
        benchmark_end = pd.DataFrame(yf.download(tickers=benchmark, start=end_date +timedelta(days=-1), end= end_date,interval='1d',progress=False)).dropna()
        change_yield = (benchmark_end.iloc[0]['Open'] -benchmark_start.iloc[0]['Open'])/ benchmark_start.iloc[0]['Open']
        change_yield =MULTIPLY*100*change_yield
        for combination in self.combination_generator(generator_list):
            self.reset_backtest_strategy()
            group = []
            for rule in combination:
                group.append(rule)
                exec(rule)
            change = self.start_backtest_strategy(start_date,end_date)*100
            group = [change]+group
            if change > change_yield:
                iteration_count += 1
                good_combinations.append(self.change_results_to_text(group))

            if iteration_count % save_interval == 0:
                with open(file_path, 'wb') as file:
                    pickle.dump(good_combinations, file)
        
        with open(file_path, 'wb') as file:
            # pickle.dump(self.change_relutes_to_text(good_combinations), file)
            pickle.dump(good_combinations, file)

    def change_results_to_text(self,group):
        group_only_lambda =[]
        for element in group:
            if type(element) == str:
                element = element.split('(')[1]
                # element = element.split(')')[0]
                group_only_lambda.append('('+element)
            else: group_only_lambda.append(element)       
        return group_only_lambda
        


#############################################################################################################################################
############################################################## END BACKTEST DF ##############################################################
#############################################################################################################################################


if __name__=="__main__":
    time_start=datetime.now(tz=TIMEZONE)
    try:
        benchmarks = ['SPY']
        trading_strategy = TradingStrategy(verbose=False)
        trading_strategy.add_rule_buy_sell(lambda row: row['sma10'] < row['sma5'] and row['vix_sma10'] > row['vix_sma5'],type='buy') 
        trading_strategy.add_rule_buy_sell(lambda row: row['daily_macd_diff'] > -0.5,type='buy',range=[-0.5,0,0.5,1])
        trading_strategy.add_rule_buy_sell(lambda row: row['weekly_macd_diff'] > -0.5,type='buy',range=[-0.5,0,0.5,1])
        trading_strategy.add_rule_buy_sell(lambda row: row['monthly_macd_diff'] > -0.5,type='buy',range=[-0.5,0,0.5,1])
        trading_strategy.add_rule_buy_sell(lambda row: row['rsi_daily'] > 30 and row['rsi_daily_sma'] < row['rsi_daily'],type='buy',range=[30,40,50])
        trading_strategy.add_rule_buy_sell(lambda row: row['rsi_monthly'] > 30 and row['rsi_monthly_sma'] < row['rsi_monthly'],type='buy',range=[30,40,50])
        trading_strategy.add_rule_buy_sell(lambda row: row['trend_to_god_week'] == 1,type='buy')
        trading_strategy.add_rule_buy_sell(lambda row: row['daily_below_lower_band'] == 1,type='buy')
        trading_strategy.add_rule_buy_sell(lambda row: row['cc'] == 1,type='buy')
        trading_strategy.add_rule_buy_sell(lambda row: row['dew'] == 1,type='buy')
        trading_strategy.add_rule_buy_sell(lambda row: row['last_year_change'] <= -15,type='buy',range={'last_year_change':[-25,-20,-15]})
        trading_strategy.add_rule_buy_sell(lambda row: row['trend_to_god_month'] == 1,type='buy')

        trading_strategy.add_rule_buy_sell(lambda row: row['vix'] > 23 and row['vix_sma10'] < row['vix_sma5'],type='sell',weight=3,range={'vix':[22,23,24]}) # not defulte weight 
        trading_strategy.add_rule_buy_sell(lambda row: row['daily_macd_diff'] < -0.5 and row['weekly_macd_diff'] < -0.5,type='sell',range={'daily_macd_diff':[-0.5,0,0.5,1],'weekly_macd_diff':[-0.5,0,0.5,1]})
        trading_strategy.add_rule_buy_sell(lambda row: row['weekly_macd_diff'] < -0.5,type='sell',range=[-0.5,0,0.5,1])
        trading_strategy.add_rule_buy_sell(lambda row: row['monthly_macd_diff'] < 0,type='sell',range=[-0.5,0,0.5,1])
        trading_strategy.add_rule_buy_sell(lambda row: row['rsi_weekly'] < 50 and row['rsi_weekly_sma'] > row['rsi_weekly'],type='sell',range={'rsi_weekly':[40,50,60]})
        trading_strategy.add_rule_buy_sell(lambda row: row['rsi_monthly'] < 50 and row['rsi_monthly_sma'] > row['rsi_monthly'],type='sell',range={'rsi_monthly':[40,50,60]})
        trading_strategy.add_rule_buy_sell(lambda row: row['trend_to_god_week'] == 0,type='sell')
        trading_strategy.add_rule_buy_sell(lambda row: row['trend_to_god_month'] == 0,type='sell',weight=3)

        trading_strategy.set_ccrash_conditions(lambda row:row['sma5']<row['sma10'] and row['vix_sma5'] > row['vix_sma10'] and row['vix']>22) # defulte
        trading_strategy.set_thresholds_exposures(thresholds =[0.4,0.5,0.6,0.7,0.8], target_exposures=[1.5,1.2,1,0.7,0.5]) # defulte


        for benchmark in benchmarks:
            trading_strategy.benchmark = benchmark
            trading_strategy.build_dataframe()
            trading_strategy.run_backtest(start_date='2018-01-01',verbose=True)
            # trading_strategy.optimizer(start_date='2018-01-01',optimize_weight=False,optimize_val=True,estimated_runtime=True)
        total_seconds = (datetime.now(tz=TIMEZONE) - time_start).total_seconds()
        # se.send_me_email(f"End checking weights combinations for {benchmark} in {total_seconds/60} Minutes - {PROGRAM_NAME}")

    except Exception as e:
        if hasattr(e, "line"):
            line = e.line
        else:
            line = "unknown"
        if hasattr(e, "severity"):
            severity = e.severity
        else:
            severity = "ERROR"
        if hasattr(e, "message"):
            message = e.message
        else:
            message = traceback.format_exc()
        total_seconds = (datetime.now(tz=TIMEZONE) - time_start).total_seconds()
        send_message = f"message: {message}\n, line: {line}\n,app: {PROGRAM_NAME} \n,severity: ERROR \nFaild in {total_seconds/60} minuts"
        se.send_me_email(send_message)
        # print(send_message)
        # logger.log_struct({"message": message, "line": line ,"app":PROGRAM_NAME ,"severity": severity})
