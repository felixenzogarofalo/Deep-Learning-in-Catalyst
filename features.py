import pandas as pd
import numpy as np
from ta import *
from scipy import stats
import scipy.optimize
from scipy.optimize import OptimizeWarning
import warnings
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# from matplotlib.finance import _candlestick
from matplotlib.dates import date2num
from datetime import datetime

class holder():
    1

# Heiken Ashi Candles
def heikenashi(prices):
    """
    param prices: dataframe de datos OHLC y volumen
    param periods: periodos para los cuales se crearan las velas
    return: velas heiken ashi OHLC
    """
    HAclose = prices[["open", "high", "low", "close"]].sum(axis=1) / 4
    HAopen = HAclose.copy()
    HAopen.iloc[0] = HAclose.iloc[0]

    HAhigh = HAclose.copy()
    HAlow = HAclose.copy()

    for i in range(1, len(prices)):
        HAopen.iloc[i] = (HAopen.iloc[i - 1] + HAclose.iloc[i - 1]) / 2
        HAhigh.iloc[i] = np.array((prices["high"].iloc[i], HAopen.iloc[i], HAclose.iloc[i])).max()
        HAlow.iloc[i] = np.array((prices["low"].iloc[i], HAopen.iloc[i], HAclose.iloc[i])).min()

    df = pd.concat((HAopen, HAhigh, HAlow, HAclose), axis=1)
    df.columns = ["HAopen", "HAhigh", "HAlow", "HAclose"]

    return df


# Quitar tendencia
def detrend(prices, method="difference"):
    """
    :param prices: dataframe de datos OHLC y volumen
    :param method: cadena de texto con el método a utilizar
    :return: pandas DataFrame con datos de cierre sin tendencia
    """
    if method == "difference":
        detrended = pd.DataFrame(data=np.zeros(len(prices)))
        detrended.iloc[0] = prices.close[1] - prices.close[0]
        for i in range(1, len(prices)):
            detrended.iloc[i] = prices.close[i] - prices.close[i-1]
        detrended = detrended.values

    elif method == "linear":
        x = np.arange(0, len(prices))
        y = prices["close"].values

        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y.reshape(-1, 1))

        trend = model.predict(x.reshape(-1, 1))
        trend = trend.reshape((len(prices),))

        detrended = prices.close - trend

    else:
        print("No se ha especificado un método correcto para quitar la tendencia")
        return

    detrended = pd.DataFrame(data=detrended,
                             index=prices.index.tolist(),
                             columns=["Detrended"])

    return detrended

# función de ajuste de la Serie de Expansión de Fourier
def fseries(x, a0, a1, b1, w):
    """

    :param x: valor de tiempo
    :param a0: primer coeficiente de la serie
    :param a1: segundo coeficiente de res = scipy.optimize.curve_fit(fseries, x, y)la serie
    :param b1: tercer coeficiente de la serie
    :param w: frecuencia de la serie
    :return: retorna el valor de la serie de Fourier
    """

    f = a0 + a1 * np.cos(w * x) + b1 * np.sin(w * x)

    return f


# función de ajuste de la Serie de Expansión de Seno
def sseries(x, a0, b1, w):
    """

    :param x: valor de tiempo
    :param a0: primer coeficiente de la serie
    :param b1: tercer coeficiente de la serie
    :param w: frecuencia de la serie
    :return: retorna el valor de la serie de Fourier
    """

    f = a0 + b1 * np.sin(w * x)

    return f

# Función que calcula los coeficientes de la serie de Fourier
def fourier(prices, periods, method="difference"):
    """
    :param prices: OHLC dataframe
    :param periods: lista de periodos para los cuales computar los coeficientes
    :param method: método por el cual quitar la tendencia
    :return: diccionario de coeficientes para los períodos especificados
    """

    results = holder()
    dict = {}

    plot = False

    # Computar los coeficientes

    detrended = detrend(prices, method)

    p0 = None
    for i in range(0, len(periods)):
        coeffs = []

        for j in range(periods[i], len(prices)):
            x = np.arange(0, periods[i])
            y = detrended.iloc[j - periods[i]:j].values.reshape(-1)

            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)

                try:
                    if p0 is None:
                        res, _ = scipy.optimize.curve_fit(fseries, x, y, p0)
                        p0 = res
                    else:
                        res, _ = scipy.optimize.curve_fit(fseries, x, y, p0)
                except (RuntimeError, OptimizeWarning):
                    res = np.empty(4)
                    res[:] = np.NAN

            if plot == True:
                xt = np.linspace(0, periods[i], 100)
                yt = fseries(xt, res[0], res[1], res[2], res[3])

                plt.plot(x, y)
                plt.plot(xt, yt, "r")

                plt.show()

            coeffs = np.append(coeffs, res, axis=0)

        coeffs = np.array(coeffs).reshape((len(coeffs)//(len(res)), len(res)))

        df = pd.DataFrame(coeffs, index=prices.iloc[periods[i]:])

        df.columns = [["a0", "a1", "b1", "w"]]

        df.fillna(method="ffill")

        dict[periods[i]] = df

        results.coeffs = dict

    return results


# Función que calcula los coeficientes de la serie de Sen
def sine(prices, periods, method="difference"):
    """
    :param prices: OHLC dataframe
    :param periods: lista de periodos para los cuales computar los coeficientes
    :param method: método por el cual quitar la tendencia
    :return: diccionario de coeficientes para los períodos especificados
    """

    results = holder()
    dict = {}

    plot = False

    # Computar los coeficientes

    detrended = detrend(prices, method)

    p0 = None
    for i in range(0, len(periods)):
        coeffs = []

        for j in range(periods[i], len(prices)):
            x = np.arange(0, periods[i])
            y = detrended.iloc[j - periods[i]:j].values.reshape(-1)

            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)

                try:
                    res, _ = scipy.optimize.curve_fit(sseries, x, y, method="lm", maxfev=800)
                except (RuntimeError, OptimizeWarning):
                    res = np.empty(3)
                    res[:] = np.NAN
            if plot == True:
                xt = np.linspace(0, periods[i], 100)
                yt = sseries(xt, res[0], res[1], res[2])

                plt.plot(x, y)
                plt.plot(xt, yt, "r")

                plt.show()

            coeffs = np.append(coeffs, res, axis=0)

        coeffs = np.array(coeffs).reshape((len(coeffs)//(len(res)), len(res)))

        df = pd.DataFrame(coeffs, index=prices.iloc[periods[i]:])
        df.columns = [["a0", "b1", "w"]]

        df.fillna(method="ffill")
        dict[periods[i]] = df

    results.coeffs = dict

    return results

def wadl(prices, period):
    """
    Williams Accumulation Distribution Function
    :param prices: dataframe de precios OHLC
    :param period: (list) período para calcular la función
    :return: Lineas de Williams Accumulation Distribution para cada período
    """

    results = holder()
    dict = {}

    WAD = []

    for j in range(period, len(prices)):
        TRH = np.array([prices.high.iloc[j], prices.close.iloc[j-1]]).max()
        TRL = np.array([prices.low.iloc[j], prices.close.iloc[j - 1]]).min()

        if prices.close.iloc[j] > prices.close.iloc[j-1]:
            PM = prices.close.iloc[j] - TRL
        elif prices.close.iloc[j] < prices.close.iloc[j-1]:
            PM = prices.close.iloc[j] - TRH
        elif prices.close.iloc[j] == prices.close.iloc[j-1]:
            PM = 0
        else:
            print(str(prices.close.iloc[j]), str(prices.close.iloc[j-1]))
            print(prices.shape)
            print("Error inesperado")

        AD = PM * prices.volume.iloc[j]

        WAD = np.append(WAD, AD)

    WAD = WAD.cumsum().reshape(-1, 1)

    array = np.empty(shape=(prices.shape[0],))
    array[:] = np.nan
    WADL = pd.DataFrame(data=array,
                        index=prices.index.tolist(),
                        columns=["WAD"])

    WADL.iloc[period:] = WAD
    WADL.fillna(method="bfill")

    return WADL

def create_up_down_dataframe(data,
                             lookfoward_w=5,
                             up_down_factor=2.0,
                             percent_factor=0.01):
    """
    Crea un DataFrame de pandas que crea un etiqueta cuando el mercado se mueve hacia arriba
    "up_down_factor * percent_factor" en período de "lookfoward_w" mientras que no cae por debajo de "percent_factor"
    en el mismo período
    :param data: DataFrame con los datos
    :param lookback_w: ventana para mirar hacia atrás
    :param lookfoward_w: venta de predicción
    :param up_down_factor: factor de amplificación
    :param percent_factor: porcentaje
    :return: DataFrame con la característica descrita previamente
    """
    data = data.copy()
    # Eliminar las columnas que no son escenciales
    data.drop(["open", "low", "high", "volume"], axis=1, inplace=True)

    # Crear los desplazamiento hacia hacia adelante
    for i in range(lookfoward_w):
        data["Lookfoward%s" % str(i+1)] = data["close"].shift(-(i+1))

    data.fillna(method="ffill")

    # Ajustar todos estos valores para que sean porcentajes de retorno
    for i in range(lookfoward_w):
        data["Lookfoward%s" % str(i+1)] = data["Lookfoward%s" % str(i+1)].pct_change() * 100.0

    data.fillna(method="ffill")

    direction = up_down_factor * percent_factor
    opposite = percent_factor

    # Crear una lista de Verdadero/Falso para cada fecha en que se cumple la lógica expuesta en la descripción
    down_up_cols = [data["Lookfoward%s" % str(i+1)] > -opposite
             for i in range(lookfoward_w)]
    up_up_cols = [data["Lookfoward%s" % str(i+1)] > direction
             for i in range(lookfoward_w)]
    up_down_cols = [data["Lookfoward%s" % str(i+1)] < opposite
             for i in range(lookfoward_w)]
    down_down_cols = [data["Lookfoward%s" % str(i+1)] < -direction
             for i in range(lookfoward_w)]

    down_up_tot = down_up_cols[0]
    for c in down_up_cols[1:]:
        down_up_tot = down_up_tot & c
    up_up_tot = up_up_cols[0]
    for c in up_up_cols[1:]:
        up_up_tot = up_up_tot | c

    up = down_up_tot & up_up_tot

    up_down_tot = up_down_cols[0]
    for c in up_down_cols[1:]:
        up_down_tot = up_down_tot & c
    down_down_tot = down_down_cols[0]
    for c in down_down_cols[1:]:
        down_down_tot = down_down_tot | c

    down = up_down_tot & down_down_tot
    up_down_label = pd.DataFrame(data=np.zeros(shape=len(up), dtype=np.int32), index=data.index, columns=["UpDown"])
    # Crear etiqueta con lógica expuesta
    for i in range(len(up)):
        if up[i]:
            up_down_label.iloc[i] = 0
        elif down[i]:
            up_down_label.iloc[i] = 1
        else:
            up_down_label.iloc[i] = 2
    """
    data["UpDown"] = up
    data["UpDown"] = data["UpDown"].astype(int)
    data["UpDown"].replace(to_replace=0, value=-1, inplace=True)
    """
    return up_down_label

def bollinger_bands(prices, k=2, period=20):
    close = prices.close
    sma = close.rolling(window=period).mean()
    rstd = close.rolling(window=20).std()

    upper_band = sma + k * rstd
    upper_band = upper_band.rename(columns=["upper_bb"])
    lower_band = sma - k * rstd
    lower_band = lower_band.rename(columns=["lower_bb"])

    df = upper_band.join(lower_band)

    return df

