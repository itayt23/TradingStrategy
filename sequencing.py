import yfinance as yf
import pandas as pd
from datetime import timedelta
import calendar
import pandas as pd




class SequenceMethodAll:

    def __init__(self,ticker,start_date=None,stop_date=None):
        if stop_date is None:
            self.symbol_data_1d = pd.DataFrame(yf.download(ticker, period='max',interval='1d',progress=False,rounding=True)).dropna()
            self.symbol_data_1d = self.symbol_data_1d.rename_axis('Date').reset_index()
            self.symbol_data_1wk = pd.DataFrame(yf.download(ticker, period='max',interval='1wk',progress=False,rounding=True)).dropna()
            self.symbol_data_1wk = self.symbol_data_1wk.rename_axis('Date').reset_index()
            self.symbol_data_1mo = pd.DataFrame(yf.download(ticker, period='max',interval='1mo',progress=False,rounding=True)).dropna()
            self.symbol_data_1mo = self.symbol_data_1mo.rename_axis('Date').reset_index()
        else:
            self.symbol_data_1d = pd.DataFrame(yf.download(ticker, start=start_date, end=stop_date,interval='1d',progress=False,rounding=True)).dropna()
            self.symbol_data_1d = self.symbol_data_1d.rename_axis('Date').reset_index()
            self.symbol_data_1wk = pd.DataFrame(yf.download(ticker, start=start_date, end=stop_date,interval='1wk',progress=False,rounding=True)).dropna()
            self.symbol_data_1wk = self.symbol_data_1wk.rename_axis('Date').reset_index()
            self.symbol_data_1mo = pd.DataFrame(yf.download(ticker, start=start_date, end=stop_date,interval='1mo',progress=False,rounding=True)).dropna()
            self.symbol_data_1mo = self.symbol_data_1mo.rename_axis('Date').reset_index()
        self.sequence_daily = pd.DataFrame(columns=['Date', 'Entry Price', 'Sequence', 'Days', "Yield"])
        self.sequence_weekly = pd.DataFrame(columns=['Date', 'Entry Price', 'Sequence', 'Days', "Yield"])
        self.sequence_monthly = pd.DataFrame(columns=['Date', 'Entry Price', 'Sequence', 'Days', "Yield"])
        self.sequence_daily,self.sequence_weekly,self.sequence_monthly = build_sequences(self)

    def get_last_daily_seq_days_number(self):
        return self.sequence_daily['Days'].iloc[-1]
    
    def get_last_weekly_seq_days_number(self):
        return self.sequence_weekly['Days'].iloc[-1]
    
    def get_weekly_seq_days_number_by_date(self,date):
        closest_date = min(self.sequence_weekly['Date'], key=lambda x: abs(pd.Timestamp(x) - pd.Timestamp(date)))
        if pd.Timestamp(closest_date) > pd.Timestamp(date):
            closest_date = self.sequence_weekly['Date'][self.sequence_weekly['Date'].get_loc(closest_date)-1]
        return self.sequence_weekly[self.sequence_weekly['Date'] == closest_date]['Days'].iloc[0]

    def get_last_monthly_seq_days_number(self):
        return self.sequence_monthly['Days'].iloc[-1]
    
    def get_monthly_seq_days_number_by_date(self,date):
        closest_date = min(self.sequence_monthly['Date'], key=lambda x: abs(pd.Timestamp(x) - pd.Timestamp(date)))
        if pd.Timestamp(closest_date) > pd.Timestamp(date):
            closest_date = self.sequence_monthly['Date'][self.sequence_monthly['Date'].get_loc(closest_date)-1]
        return self.sequence_monthly[self.sequence_monthly['Date'] == closest_date]['Days'].iloc[0]
        
    def get_daily_seq_df(self):
        return self.sequence_daily
    
    def get_weekly_seq_df(self):
        return self.sequence_weekly
    
    def get_monthly_seq_df(self):
        return self.sequence_monthly
    
    def get_last_sequence_daily(self):
        return self.sequence_daily['Sequence'].iloc[-1]
    
    def get_last_sequence_weekly(self):
        return self.sequence_weekly['Sequence'].iloc[-1]
    
    def get_sequence_weekly_by_date(self,date):
        closest_date = min(self.sequence_weekly['Date'], key=lambda x: abs(pd.Timestamp(x) - pd.Timestamp(date)))
        if pd.Timestamp(closest_date) > pd.Timestamp(date):
            closest_date = self.sequence_weekly['Date'][self.sequence_weekly['Date'].get_loc(closest_date)-1]
        return self.sequence_weekly[self.sequence_weekly['Date'] == closest_date]['Sequence'].iloc[0]
    
    def get_last_sequence_monthly(self):
        return self.sequence_monthly['Sequence'].iloc[-1]
    
    def get_sequence_monthly_by_date(self,date):
        closest_date = min(self.sequence_monthly['Date'], key=lambda x: abs(pd.Timestamp(x) - pd.Timestamp(date)))
        if pd.Timestamp(closest_date) > pd.Timestamp(date):
            closest_date = self.sequence_monthly['Date'][self.sequence_monthly['Date'].get_loc(closest_date)-1]
        return self.sequence_monthly[self.sequence_monthly['Date'] == closest_date]['Sequence'].iloc[0]

    def get_avg_up_return(self, interval):
        sequence = pd.DataFrame()
        counter = 0
        seq_yield = 0
        avg_yield = 0
        if(interval == 'day'): sequence = self.sequence_daily
        elif(interval == 'week'): sequence = self.sequence_weekly
        elif(interval == 'month'): sequence = self.sequence_monthly

        for index,row in sequence.iterrows():
            if(row["Sequence"] == 1):
                if(row["Yield"] > 0):
                    counter = counter + 1
                    seq_yield = seq_yield + row["Yield"]
        
        if counter != 0 :  avg_yield = seq_yield / counter
        return avg_yield

    def get_avg_down_return(self, interval):
        sequence = pd.DataFrame()
        counter = 0
        seq_yield = 0
        avg_yield = 0
        if(interval == 'day'): sequence = self.sequence_daily
        elif(interval == 'week'): sequence = self.sequence_weekly
        elif(interval == 'month'): sequence = self.sequence_monthly

        for index,row in sequence.iterrows():
            if(row["Sequence"] == -1):
                if(row["Yield"] < 0):
                    counter = counter + 1
                    seq_yield = seq_yield + row["Yield"]
        if counter != 0 : avg_yield = seq_yield / counter
        return avg_yield


def build_sequences(self):
    seq_1d = pd.DataFrame
    seq_1wk = pd.DataFrame
    seq_1mo = pd.DataFrame
    for interval in range(0,3):
        if(interval == 0): symbol_df = self.symbol_data_1d
        elif(interval == 1): symbol_df = self.symbol_data_1wk
        elif(interval == 2): symbol_df = self.symbol_data_1mo
        sequence = pd.DataFrame(columns=['Date', 'Sequence', 'Days', "Yield"])
        seq_df_index = 0
        down_seq = up_seq = False
        down_seq_list = up_seq_list =[]
        close_price = low_price = high_price =  enter_price = sell_price = seq_yield = 1
        days = 0
        in_sequence = False
        first = True
        for i in range(len(symbol_df)):
            if first:
                first = False
                continue
        
            close_price = symbol_df.loc[i, "Close"]
            low_price = symbol_df.loc[i-1, "Low"]
            high_price = symbol_df.loc[i-1, "High"]
            date = symbol_df.loc[i, "Date"].date()
            if(interval == 2):
                date = date.replace(day = calendar.monthrange(date.year, date.month)[1])
            if(interval == 1):
                start = date - timedelta(days=date.weekday())
                date = start + timedelta(days=6)
            if not in_sequence:
                if(close_price > low_price):
                    up_seq = True
                    up_seq_list.append((high_price,low_price))
                    up_seq_list.append((symbol_df.loc[i, "High"],symbol_df.loc[i, "Low"]))
                    sequence.loc[seq_df_index,'Date'] = date
                    sequence.loc[seq_df_index,'Sequence'] = 1
                    sequence.loc[seq_df_index,'Entry Price'] = symbol_df.loc[i, "Low"]
                    enter_price = symbol_df.loc[i, "Close"]
                    in_sequence = True
                elif(close_price < high_price):
                    down_seq = True
                    down_seq_list.append((low_price,high_price))
                    down_seq_list.append((symbol_df.loc[i, "Low"],symbol_df.loc[i, "High"]))
                    sequence.loc[seq_df_index,'Date'] = date
                    sequence.loc[seq_df_index,'Sequence'] = -1
                    sequence.loc[seq_df_index,'Entry Price'] = symbol_df.loc[i, "High"]
                    sell_price = symbol_df.loc[i, "Close"]
                    in_sequence = True
            else:
                if(up_seq):
                    days = days + 1
                    sequence.loc[seq_df_index,'Days'] = days
                    if(close_price > max(up_seq_list)[1]):
                        up_seq_list.append((symbol_df.loc[i, "High"],symbol_df.loc[i, "Low"]))
                        continue
                    else: #finsih up trend and starting down trend
                        sell_price = symbol_df.loc[i, "Close"]
                        seq_yield = (sell_price - enter_price)/enter_price*100
                        sequence.loc[seq_df_index,'Yield'] = seq_yield
                        sequence.loc[seq_df_index,'Days'] = days
                        days = seq_yield = 0
                        up_seq_list = down_seq_list =[]
                        up_seq = False
                        down_seq = True
                        seq_df_index = seq_df_index + 1
                        down_seq_list.append((symbol_df.loc[i, "Low"],symbol_df.loc[i, "High"]))
                        sequence.loc[seq_df_index,'Date'] = date
                        sequence.loc[seq_df_index,'Sequence'] = -1
                        sequence.loc[seq_df_index,'Entry Price'] = symbol_df.loc[i, "High"]
                        enter_price = symbol_df.loc[i, "Close"]
                elif(down_seq):
                    days = days + 1
                    sequence.loc[seq_df_index,'Days'] = days
                    if(close_price < min(down_seq_list)[1]):
                        down_seq_list.append((symbol_df.loc[i, "Low"],symbol_df.loc[i, "High"]))
                        continue
                    else: #turning from dowm trend to up trend
                        sell_price = symbol_df.loc[i, "Close"]
                        seq_yield = (sell_price - enter_price)/enter_price*100
                        sequence.loc[seq_df_index,'Yield'] = seq_yield*(-1)
                        sequence.loc[seq_df_index,'Days'] = days
                        days = seq_yield = 0
                        up_seq_list = down_seq_list =[]
                        down_seq = False
                        up_seq = True
                        seq_df_index = seq_df_index + 1
                        up_seq_list.append((symbol_df.loc[i, "High"],symbol_df.loc[i, "Low"]))
                        sequence.loc[seq_df_index,'Date'] = date
                        sequence.loc[seq_df_index,'Sequence'] = 1
                        sequence.loc[seq_df_index,'Entry Price'] = symbol_df.loc[i, "Low"]
                        enter_price = symbol_df.loc[i, "Close"]
        if(interval == 0):
            seq_1d = sequence
        if(interval == 1):
            seq_1wk = sequence
        if(interval == 2):
            seq_1mo = sequence
    return seq_1d,seq_1wk,seq_1mo





class SequenceMethod:

    def __init__(self, data, interval,stop_date):
        self.data = data
        self.interval = interval
        self.sequence = get_sequence(data,interval,stop_date)

    def get_last_sequence(self):
        return self.sequence['Sequence'].iloc[-1]

    def get_seq_df(self):
        return self.sequence

    def get_avg_up_return(self):
        counter = 0
        seq_yield = 0
        avg_yield = 0
        seq_df = self.sequence.dropna()
        for index,row in seq_df.iterrows():
            if(row["Sequence"] == 1):
                if(row["Yield"] > 0):
                    counter = counter + 1
                    seq_yield = seq_yield + row["Yield"]
        try:
            avg_yield = seq_yield / counter
        except: return 7
        return avg_yield

    def get_avg_down_return(self):
        counter = 0
        seq_yield = 0
        avg_yield = 0
        for index,row in self.sequence.iterrows():
            if(row["Sequence"] == -1):
                if(row["Yield"] < 0):
                    counter = counter + 1
                    seq_yield = seq_yield + row["Yield"]
        try:
            avg_yield = seq_yield / counter
        except: return 7
        return avg_yield
    def get_last_number_of_seq_candels(self):
        return self.sequence['Days'].iloc[-1]

def get_sequence(data, interval,stop_date):
    symbol_df = data.rename_axis('Date').reset_index()
    sequence = pd.DataFrame(columns=['Date', 'Entry Price', 'Sequence', 'Days', "Yield"])
    seq_df_index = 0
    down_seq = up_seq = False
    down_seq_list = up_seq_list =[]
    close_price = low_price = high_price =  enter_price = sell_price = seq_yield = 1
    days = 0
    in_sequence = False
    first = True
    for i in range(len(symbol_df)):
        if first:
            first = False
            continue
        close_price = symbol_df.loc[i, "Close"]
        low_price = symbol_df.loc[i-1, "Low"]
        high_price = symbol_df.loc[i-1, "High"]
        date = symbol_df.loc[i, "Date"].date()
        if(interval == 'monthly'):
            date = date.replace(day = calendar.monthrange(date.year, date.month)[1])
        if(interval == 'weekly'):
            start = date - timedelta(days=date.weekday())
            date = start + timedelta(days=6)
        if(date >= stop_date):
            break
        if not in_sequence:
            if(close_price > low_price):
                up_seq = True
                up_seq_list.append((high_price,low_price))
                up_seq_list.append((symbol_df.loc[i, "High"],symbol_df.loc[i, "Low"]))
                sequence.loc[seq_df_index,'Date'] = date
                sequence.loc[seq_df_index,'Sequence'] = 1
                sequence.loc[seq_df_index,'Entry Price'] = symbol_df.loc[i, "Low"]
                enter_price = symbol_df.loc[i, "Close"]
                in_sequence = True
            elif(close_price < high_price):
                down_seq = True
                down_seq_list.append((low_price,high_price))
                down_seq_list.append((symbol_df.loc[i, "Low"],symbol_df.loc[i, "High"]))
                sequence.loc[seq_df_index,'Date'] = date
                sequence.loc[seq_df_index,'Sequence'] = -1
                sequence.loc[seq_df_index,'Entry Price'] = symbol_df.loc[i, "High"]
                sell_price = symbol_df.loc[i, "Close"]
                in_sequence = True
        else:
            if(up_seq):
                days = days + 1
                sequence.loc[seq_df_index,'Days'] = days #! Added it but it is not suppused to affect the logic beore
                if(close_price >= max(up_seq_list)[1]):
                    up_seq_list.append((symbol_df.loc[i, "High"],symbol_df.loc[i, "Low"]))
                    continue
                else: #finsih up trend and starting down trend
                    sell_price = symbol_df.loc[i, "Close"]
                    seq_yield = (sell_price - enter_price)/enter_price*100
                    sequence.loc[seq_df_index,'Yield'] = seq_yield
                    sequence.loc[seq_df_index,'Days'] = days
                    days = seq_yield = 0
                    up_seq_list = down_seq_list =[]
                    up_seq = False
                    down_seq = True
                    seq_df_index = seq_df_index + 1
                    down_seq_list.append((symbol_df.loc[i, "Low"],symbol_df.loc[i, "High"]))
                    sequence.loc[seq_df_index,'Date'] = date
                    sequence.loc[seq_df_index,'Sequence'] = -1
                    sequence.loc[seq_df_index,'Entry Price'] = symbol_df.loc[i, "High"]
                    enter_price = symbol_df.loc[i, "Close"]
            elif(down_seq):
                days = days + 1
                sequence.loc[seq_df_index,'Days'] = days #! Added it but it is not suppused to affect the logic beore
                if(close_price <= min(down_seq_list)[1]):
                    down_seq_list.append((symbol_df.loc[i, "Low"],symbol_df.loc[i, "High"]))
                    continue
                else: #turning from dowm trend to up trend
                    sell_price = symbol_df.loc[i, "Close"]
                    seq_yield = (sell_price - enter_price)/enter_price*100
                    sequence.loc[seq_df_index,'Yield'] = seq_yield*(-1)
                    sequence.loc[seq_df_index,'Days'] = days
                    days = seq_yield = 0
                    up_seq_list = down_seq_list =[]
                    down_seq = False
                    up_seq = True
                    seq_df_index = seq_df_index + 1
                    up_seq_list.append((symbol_df.loc[i, "High"],symbol_df.loc[i, "Low"]))
                    sequence.loc[seq_df_index,'Date'] = date
                    sequence.loc[seq_df_index,'Sequence'] = 1
                    sequence.loc[seq_df_index,'Entry Price'] = symbol_df.loc[i, "Low"]
                    enter_price = symbol_df.loc[i, "Close"]

    return sequence


   