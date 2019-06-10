from catalyst import run_algorithm
from catalyst.api import symbols
import matplotlib.pyplot as plt
from synthetic_data import SyntheticData
import tensorflow as tf
import pandas as pd
from logbook import Logger
from sklearn import preprocessing
import pickle

NAMESPACE = 'Deep Deterministic Policy Gradient'
log = Logger(NAMESPACE)


def initialize(context):
    # Lista de activos
    context.assets = ["btc_usdt"]
    context.symbols = symbols("btc_usdt")

    # Definir carácteristicas
    context.row_features = ["open", "high", "low", "close"]
    context.features = ["open", "high", "low", "close"]

    # Activar siguiente línea para inlcuir velas Heiken Ashi
    context.include_ha = True
    if context.include_ha:
        context.features.append("HAopen")
        context.features.append("HAhigh")
        context.features.append("HAlow")
        context.features.append("HAclose")

    # Activar siguiente línea para incluir Distribución de Acumulación de William
    context.include_wadl = False
    if context.include_wadl:
        context.features.append("WAD")

    context.render = True

    context.min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))


def handle_data(context, data):
    if context.render:
        context.synthetic_data = SyntheticData(context=context,
                                               data=data,
                                               window=10000,
                                               frequency=30)
        data, close_prices = context.synthetic_data.get_trayectory(t_intervals=1000 + 50)

        plt.plot(data[0, :, 0])
        plt.show()


    context.render = False



def analyze(context, perf):
    pass


if __name__ == '__main__':
    run_algorithm(
        capital_base=1000,
        data_frequency='minute',
        initialize=initialize,
        handle_data=handle_data,
        analyze=analyze,
        exchange_name='poloniex',
        algo_namespace=NAMESPACE,
        quote_currency='usd',
        start=pd.to_datetime('2018-1-1', utc=True),
        end=pd.to_datetime('2018-1-1', utc=True),
    )
