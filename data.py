import pandas_datareader.data as web
import yfinance as yf
yf.pdr_override()
import pandas as pd
import numpy as np

class Data():

    def __init__(self, ticker, start, end):

        self.ticker = ticker
        self.start = start
        self.end = end
        self.data = self.get_data()

    def get_data(self):
        """downloads daily stock price data from yahoo finance for given ticker symbol from start to end data

        Returns:
            dataframe: the downloaded data
        """
        df = web.get_data_yahoo(self.ticker, start=self.start, end=self.end)
        return df

    def prep_data(self, n):
        """preps data such that technical indicators are added

        Args:
            n (int): number of days used for calculation of technical indicators
        """
        self.data['MACD'] = self.cal_macd()
        self.data['MA'] = self.cal_ma(n)
        self.data['EMA'] = self.cal_ema(n)
        self.data['ATR'] = self.cal_atr(n)
        self.data['ROC'] = self.cal_roc(n)

    def cal_macd(self):
        """calculates the moving average convergence divergence

        Returns:
            series: the metric mentioned
        """
        return self.cal_ema(12) - self.cal_ema(26)

    def cal_ma(self, n):
        """calculates the simple moving average

        Args:
            n (int): number of days used for the average

        Returns:
            series: returns metric as mentioned for given data
        """
        return self.data.Close.rolling(window=n, min_periods=1).mean()

    def cal_ema(self, n):
        """calculates the ema (exponential moving average)

        Args:
            n (int): number of days used

        Returns:
            series: caluclated ema for given close prices
        """
        return self.data.Close.ewm(span=n, adjust=False).mean()

    def cal_atr(self, n):
        """calculates the atr (average true range) for given data

        Args:
            n (int): number of days for calculating the average of the daily true ranges

        Returns:
            series: atr
        """
        high_min_low = self.data['High'] - self.data['Low']
        high_min_prev_close = self.data['High'] - self.data['Close'].shift(1)
        low_min_prev_close = self.data['Low'] - self.data['Close'].shift(1)
        temp = pd.concat([high_min_low, high_min_prev_close, low_min_prev_close], axis = 1)
        true_range = np.max(temp, axis = 1)
        atr = true_range.rolling(n).sum()/n
        return atr

    def cal_roc(self, n):
        """calculates the technical indicator rate of change

        Args:
            n (int): number of days used for calculating the ROC

        Returns:
            series: contains calculated ROC for given data
        """
        close_today = self.data['Close'].diff(n)
        close_n_bef = self.data['Close'].shift(n)
        roc = pd.Series((close_today/close_n_bef), name='ROC')
        return roc