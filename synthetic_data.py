import numpy as np
import pandas as ped
from scipy.stats import norm
from sklearn import preprocessing
import ta

from features import *

class SyntheticData(object):
    def __init__(self, context, data, window, frequency):
        self.context = context
        self.data = data
        self.window = window
        self.frequency = frequency
        # Crear ventana de datos.
        self.h1 = data.history(self.context.symbols,
                               self.context.row_features,
                               bar_count=self.window * self.frequency,
                               frequency="1T",
                               )

        self.h1 = self.h1.swapaxes(2, 0)

        self.close = []
        self.log_returns = []
        self.mu = []
        self.var = []
        self.drift = []
        self.stdev = []

        for i, asset in enumerate(context.assets):
            close = self.h1.iloc[i].close
            self.close.append(np.array(close))
            log_returns = np.log(1 + close.pct_change())
            self.log_returns.append(np.array(log_returns))
            mu = log_returns.mean()
            self.mu.append(mu)
            var = log_returns.var()
            self.var.append(var)
            drift = mu - (0.5 * var)
            self.drift.append(drift)
            stdev = log_returns.std()
            self.stdev.append(stdev)

        self.close = np.array(self.close)
        self.log_returns = np.array(self.log_returns)
        self.mu = np.array(self.mu)
        self.var = np.array(self.var)
        self.drift = np.array(self.drift)
        self.stdev = np.array(self.stdev)

        self.assets = []

    def get_trayectory(self, t_intervals):
        """
        :param t_intervals: número de intervalos en cada trayectoria
        :return: Datos con características de la trayectoria sintética y precios de cierre en bruto de al misma
        """
        trayectories = []
        closes = []
        p = True
        for i, asset in enumerate(self.context.assets):
            synthetic_return = np.exp(
                self.drift[i] + self.stdev[i] * norm.ppf(np.random.rand((t_intervals * self.frequency) + self.frequency, 1)))
            initial_close = self.close[i, -1]
            synthetic_close = np.zeros_like(synthetic_return)
            synthetic_close[0] = initial_close

            for t in range(1, synthetic_return.shape[0]):
                synthetic_close[t] = synthetic_close[t - 1] * synthetic_return[t]

            OHLC = []

            for t in range(synthetic_return.shape[0]):
                if t % self.frequency == 0 and t > 0:
                    open = synthetic_close[t - self.frequency]
                    high = np.max(synthetic_close[t - self.frequency: t])
                    low = np.min(synthetic_close[t - self.frequency: t])
                    close = synthetic_close[t]

                    OHLC.append([open, high, close, low])

            data = pd.DataFrame(data=OHLC, columns=["open", "high", "low", "close"])

            close = data.close

            if self.context.include_ha:
                ha = heikenashi(data)
                data = pd.concat((data, ha), axis=1)

            for period in [3, 6, 8, 10, 15, 20]:
                data["rsi" + str(period)] = ta.rsi(data.close, n=period, fillna=True)
                data["stoch" + str(period)] = ta.stoch(data.high, data.low, data.close, n=period, fillna=True)
                data["stoch_signal" + str(period)] = ta.stoch_signal(high=data.high,
                                                                     low=data.low,
                                                                     close=data.close,
                                                                     n=period,
                                                                     d_n=3,
                                                                     fillna=True)

                data["dpo" + str(period)] = ta.dpo(close=data.close,
                                                   n=period,
                                                   fillna=True)

                data["atr" + str(period)] = ta.average_true_range(high=data.high,
                                                                  low=data.low,
                                                                  close=data.close,
                                                                  n=period,
                                                                  fillna=True)

            for period in [6, 7, 8, 9, 10]:
                data["williams" + str(period)] = ta.wr(high=data.high,
                                                       low=data.low,
                                                       close=data.close,
                                                       lbp=period,
                                                       fillna=True)
            for period in [12, 13, 14, 15]:
                data["proc" + str(period)] = ta.trix(close=data.close,
                                                     n=period,
                                                     fillna=True)

            data["macd_diff"] = ta.macd_diff(close=data.close,
                                             n_fast=15,
                                             n_slow=30,
                                             n_sign=9,
                                             fillna=True)

            data["macd_signal"] = ta.macd_signal(close=data.close,
                                                 n_fast=15,
                                                 n_slow=30,
                                                 n_sign=9,
                                                 fillna=True)

            data["bb_high_indicator"] = ta.bollinger_hband_indicator(close=data.close,
                                                                     n=15,
                                                                     ndev=2,
                                                                     fillna=True)

            data["bb_low_indicator"] = ta.bollinger_lband_indicator(close=data.close,
                                                                    n=15,
                                                                    ndev=2,
                                                                    fillna=True)

            data["dc_high_indicator"] = ta.donchian_channel_hband_indicator(close=data.close,
                                                                            n=20,
                                                                            fillna=True)

            data["dc_low_indicator"] = ta.donchian_channel_lband_indicator(close=data.close,
                                                                           n=20,
                                                                           fillna=True)

            data["ichimoku_a"] = ta.ichimoku_a(high=data.high,
                                               low=data.low,
                                               n1=9,
                                               n2=26,
                                               fillna=True)

            data.fillna(method="bfill")

            # Normalizar los valores
            for feature in data.columns:
                norm_feature = preprocessing.normalize(data[feature].values.reshape(-1, 1), axis=0)
                data[feature] = pd.DataFrame(data=norm_feature, index=data.index, columns=[feature])

            self.assets = data.columns

            trayectories.append(data.values)
            closes.append(close)

        return np.array(trayectories), np.array(closes)

    def get_data_features(self):
        return self.assets
