import numpy as np
import pandas as pd
from datetime import date, timedelta
import yfinance as yf
from sequencing import SequenceMethodAll

def tense_compute_monthly(data_monthly:pd.DataFrame,sequence:SequenceMethodAll,today=date.today()):
    tense_score_monthly = 0
    consec_counter = 0
    count_candles = sequence.get_monthly_seq_days_number_by_date(today)
    if(np.isnan(count_candles)): count_candles = 1
    else: count_candles = count_candles + 1
    if(sequence.get_sequence_monthly_by_date(today) == 1):
        if(count_candles >= 19):
            tense_score_monthly = tense_score_monthly - 0.1
        if(count_candles >= 24):
            tense_score_monthly = tense_score_monthly - 0.1
    week_day = date.today().weekday()
    if(check_end_of_month(data_monthly)): consec_green = data_monthly[['Close','Open']].tail(10)
    else: consec_green = data_monthly[['Close','Open']].tail(11).head(10)
    for i in range(len(consec_green)-1, 0, -1):
        curr_close = round(consec_green.iloc[i]['Close'],2)
        prev_close = round(consec_green.iloc[i-1]['Close'],2)
        if curr_close >= prev_close * 1.01:
            consec_counter = consec_counter + 1
        else:
            break
    if(consec_counter >= 3):
        tense_score_monthly = tense_score_monthly - 0.1
    if(consec_counter >= 4):
        tense_score_monthly = tense_score_monthly - 0.1
    if(consec_counter >= 5):
        tense_score_monthly = tense_score_monthly - 0.1
    if(tense_score_monthly < -0.4): tense_score_monthly = -0.4
    return tense_score_monthly
    
def check_end_of_month(data:pd.DataFrame):
    last_date = data.index[-1].date()
    today = date.today()
    if last_date.month != today.month:
        return True
    else:
        if today.weekday() in [5,6]:
            if today.weekday() == 6: start_week = today + timedelta(days=1)
            if today.weekday() == 5: start_week = today + timedelta(days=2)
            if start_week.month != last_date.month:
                return True
    return False
    


def tense_compute_weekly(data_weekly:pd.DataFrame,sequence:SequenceMethodAll,today=date.today()):
    tense_score_weekly = 0
    consec_counter = 0
    count_candles = sequence.get_weekly_seq_days_number_by_date(today)
    if(np.isnan(count_candles)): count_candles = 1
    else: count_candles = count_candles + 1
    if(sequence.get_sequence_weekly_by_date(today) == 1):
        if(count_candles >= 8):
            tense_score_weekly = tense_score_weekly - 0.05
        if(count_candles >= 11):
            tense_score_weekly = tense_score_weekly - 0.05
    week_day = date.today().weekday()
    if(week_day in [5,6]): consec_green = data_weekly[['Close','Open']].tail(10)
    else: consec_green = data_weekly[['Close','Open']].tail(11).head(10)
    for i in range(len(consec_green)-1, 0, -1):
        curr_close = round(consec_green.iloc[i]['Close'],2)
        prev_close = round(consec_green.iloc[i-1]['Close'],2)
        if curr_close > prev_close:
            consec_counter = consec_counter + 1
        else:
            break
    if(consec_counter >= 4):
        tense_score_weekly = tense_score_weekly - 0.1
    if(consec_counter >= 7):
        tense_score_weekly = tense_score_weekly - 0.05
    if(consec_counter >= 9):
        tense_score_weekly = tense_score_weekly - 0.05
    return tense_score_weekly
    

if __name__ == '__main__':
    today = date.today()
    symbol = 'META'
    data_monthly = pd.DataFrame(yf.download(tickers=symbol, period='5y',interval='1mo',progress=False)).dropna()
    data_weekly = pd.DataFrame(yf.download(tickers=symbol, period='5y',interval='1wk',progress=False)).dropna()
    sequencing = SequenceMethodAll(symbol)
    # sequencing = SequenceMethod(symbol,stop_date=today)
    tense_score_monthly = tense_compute_monthly(data_monthly,sequencing)
    tense_score_weekly = tense_compute_weekly(data_weekly,sequencing)
