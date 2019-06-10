import matplotlib.pyplot as plt
from features import heikenashi
import ta
from synthetic_data import SyntheticData
from features import create_up_down_dataframe as updown
import numpy as np
import pandas as pd
from logbook import Logger
from sklearn import preprocessing
import tensorflow as tf
import os

from catalyst import run_algorithm
from catalyst.api import (record, symbol, order_target_percent, order_percent, )
from catalyst.api import order, record, symbol, symbols

from model import ActorNetwork, CriticNetwork
from buffer import ReplayBuffer

NAMESPACE = 'Deep Deterministic Policy Gradient'
log = Logger(NAMESPACE)


def initialize(context):
    context.TimeSpan = 15
    context.RollingWindowSize = 50

    # Lista de activos
    context.assets = ["btc_usdt", "eth_usdt", "ltc_usdt", "xmr_usdt", "str_usdt", "xrp_usdt"]
    context.symbols = symbols("btc_usdt", "eth_usdt", "ltc_usdt", "xmr_usdt", "str_usdt", "xrp_usdt")

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

    context.f = len(context.features)
    context.m = len(context.assets)
    context.n = context.RollingWindowSize

    context.initial_value = context.portfolio.starting_cash

    # Hiper-parámetros de modelo
    context.tau = 0.001
    context.actor_learning_rate = 0.0001
    context.critic_learning_rate = 0.001
    context.minibatch_size = 64
    context.max_episodes = 50
    context.max_ep_steps = 1000
    context.total_steps = context.max_episodes * context.max_ep_steps
    context.buffer_size = context.total_steps / 50  # 1000000
    context.gamma = 0.99
    context.epsilon = 0.1
    context.epsilon_target = 0.01
    context.epsilon_decay = (context.epsilon_target / context.epsilon)**(1 / (int(context.total_steps * 0.8)))

    # Valor de portafolio de entrenamiento
    context.init_train_portfolio = context.initial_value
    context.portfolio_value_memory = []
    context.portfolio_value_memory.append(context.init_train_portfolio)
    context.train_invested_quantity = 0.0
    context.train_cash = context.init_train_portfolio
    context.invested_value = 0.0
    context.portfolio_w_memory = []
    context.init_portfolio_w = []
    for i in range(len(context.assets) + 1):
        context.init_portfolio_w.append(0.0)
    context.assets_quantity_invested = []
    for i in range(len(context.assets)):
        context.assets_quantity_invested.append(0.0)
    context.init_portfolio_w[0] = 1.0
    context.portfolio_w_memory.append(context.init_portfolio_w)
    context.transaction_cost = 0.002
    context.open_trade = False
    context.model_trained = True
    context.model_created = False
    context.model_loaded = False
    context.saver = None
    context.last_train_operation = 2
    context.model_path = "/home/enzo/PycharmProjects/DDPGPorfolioOptimization/TrainedModels/model.ckpt"

    context.i = 0
    context.bar_period = 30
    context.waiting_period = 0
    context.wait = 0
    context.current_op = 2


def handle_data(context, data):
    if context.i % context.bar_period == 0:
        # Obtener datos de entrada y agregar características
        context.input, close_prices = get_data(context, data, context.n)
        if context.model_trained:
            if context.model_created:
                if not context.model_loaded:
                    context.saver = tf.train.import_meta_graph(context.model_path + ".meta")
                    context.saver.restore(context.sess, context.model_path)
                    context.model_loaded = True
            else:
                create_model(context)
                context.model_created = True
                context.saver = tf.train.import_meta_graph(context.model_path + ".meta")
                context.saver.restore(context.sess, context.model_path)
                context.model_loaded = True
        else:
            print("DDPG INFO: Entrenando el modelo 'Deep Deterministic Policy Gradient'")
            train_model(context, data, context.max_episodes)
            context.model_trained = True
            context.model_loaded = True
            print("DDPG INFO: Modelo entrenado")

        # Realizar predicción
        context.predicted = context.actor.predict([context.input])[0]

        # Realizar operación en mercado según portafolio óptimo predicho
        if context.model_trained:
            for i, asset in enumerate(context.symbols):
                if data.can_trade(asset) and context.wait == context.waiting_period:
                    order_target_percent(asset, context.predicted[i + 1])
                    context.wait = 0
                else:
                    context.wait += 1

        record(btc=data.current(symbol(context.assets[0]), "price"))
        print(str("[" + data.current_dt.strftime("%x")),
              str(data.current_dt.strftime("%X") + "]"),
              "DDGP INFO:"
              "Predicho:", str(context.predicted),
              "Valor de Portafolio:", "%.2f" % context.portfolio.portfolio_value)
    context.i += 1


def analyze(context, perf):
    # Plot the portfolio and asset data.
    ax1 = plt.subplot(211)
    perf[['portfolio_value']].plot(ax=ax1)
    ax1.set_ylabel('Portfolio\nValue\n(USD)')
    ax2 = plt.subplot(212, sharex=ax1)
    perf.btc.plot(ax=ax2)
    ax2.set_ylabel('Precio Bitcoin\n(USD)')
    plt.show()


def get_data(context, data_, window):
    # Crear ventana de datos.

    h1 = data_.history(context.symbols,
                      context.row_features,
                      bar_count=window,
                      frequency=str(context.bar_period) + "T",
                      )

    h1 = h1.swapaxes(2, 0)

    norm_data = []
    close_prices = []

    for i, asset in enumerate(context.assets):
        data = h1.iloc[i]
        close = h1.iloc[i].close
        if context.include_ha:
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

        norm_data.append(data.values)
        close_prices.append(close)
        context.features = data.columns


    return np.array(norm_data), np.array(close_prices)

# ===========================
#   Tensorflow Summary Ops
# ===========================


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax_Value", episode_ave_max_q)
    episode_loss = tf.Variable(0.)
    tf.summary.scalar("Loss", episode_loss)

    summary_vars = [episode_reward, episode_ave_max_q, episode_loss]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================


def train_model(context, data, training_batch):
    # Crear clase de datos sintéticos
    context.synthetic_data = SyntheticData(context=context,
                                           data=data,
                                           window=10000,
                                           frequency=30)

    # Crear configuración del modelo junto con redes neuronales
    create_model(context)

    # Configurar resumen de operaciones
    summary_ops, summary_vars = build_summaries()
    writer = tf.summary.FileWriter("/home/enzo/PycharmProjects/DDPGPorfolioOptimization/summaries", context.sess.graph)

    if os.path.exists(context.model_path):
        context.saver.restore(context.sess, context.model_path)

    # Inicializar la memoria de repetición
    replay_buffer = ReplayBuffer(context.buffer_size)
    for episode in range(context.max_episodes):
        data, close_prices = context.synthetic_data.get_trayectory(t_intervals=context.max_ep_steps + context.n)

        # Resetear los valores del portafolio al inicio de cada episodio
        context.portfolio_value_memory = []
        context.portfolio_value_memory.append(context.init_train_portfolio)
        context.train_invested_quantity = 0.0
        context.assets_quantity_invested = []
        context.portfolio_w_memory = []
        context.init_portfolio_w = []
        for i in range(len(context.assets) + 1):
            context.init_portfolio_w.append(0.0)
        context.portfolio_w_memory.append(context.init_portfolio_w)
        for i in range(len(context.assets)):
            context.assets_quantity_invested.append(0.0)
        context.train_cash = context.init_train_portfolio
        context.last_train_operation = 2
        context.open_trade = False

        ep_reward = 0
        ep_ave_max_q = 0
        ep_loss = 0

        # Se resta uno para tomar el cuenta la obtención del siguiente estado
        for i in range(context.max_ep_steps - 1):
            # Obtener el estado
            s = data[:, i:i + context.n, :]

            # Aplicar un error a la acción que permita equilibrar el problema de explotación/exploración
            random = np.random.rand()
            if random > context.epsilon:
                if s.shape == (len(context.assets), context.n, len(context.features)):
                    a = context.actor.predict([s])[0]
                else:
                    print("Episodio:", episode, "Paso:", i, "La forma del estado actual es incorrecta")
                    continue
            else:
                rand_array = np.random.rand(len(context.assets) + 1)
                a = np.exp(rand_array) / np.sum(np.exp(rand_array))
            context.epsilon = context.epsilon * context.epsilon_decay

            # Siguiente estado
            s2 = data[:, i + 1: i + 1 + context.n, :]
            if not s2.shape == (len(context.assets), context.n, len(context.features)):
                print("Episodio:", episode, "Paso:", i, "La forma del siguiente estado es incorrecta")
                continue

            # Recompensa
            this_closes = close_prices[:, i + context.n]
            previous_closes = close_prices[:, i + context.n - 1]

            r = get_reward(context, this_closes, previous_closes, a)

            # Punto terminal
            if i == (context.max_ep_steps - context.n - 2):
                t = True
            else:
                t = False

            replay_buffer.add(s, a, r, t, s2)

            if replay_buffer.size() > context.minibatch_size:
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(context.minibatch_size)
                # Calcular objetivos
                target_q = context.critic.predict_target(s2_batch, context.actor.predict_target(s2_batch))
                y_i = []

                for k in range(context.minibatch_size):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + context.gamma * target_q[k])

                # Actualizar el crítico dados los objetivos
                predicted_q_value_batch = np.reshape(y_i, (context.minibatch_size, 1))
                predicted_q_value, losses, _ = context.critic.train(s_batch,
                                                                    a_batch,
                                                                    predicted_q_value_batch)


                ep_loss += np.mean(losses)
                ep_ave_max_q += np.amax(predicted_q_value)

                # Actualizar la política del actor utilizando el ejemplar de gradiente
                a_outs = context.actor.predict(s_batch)
                grads = context.critic.action_gradients(s_batch, a_outs)
                context.actor.train(s_batch, grads[0])

                # Actualizar las redes objetivo
                context.actor.update_target_network()
                context.critic.update_target_network()

            ep_reward += r

            if i == (context.max_ep_steps - 2):
                summary_str = context.sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(i),
                    summary_vars[2]: ep_loss / float(i)
                })

                writer.add_summary(summary_str, episode)
                writer.flush()

                print('| Reward: {:.5f} | Episode: {:d} | Qmax: {:.4f} | Porfolio value: {:.4f} | Epsilon: {:.5f} '.format(ep_reward,
                                                                                                          episode,
                                                                                                          (ep_ave_max_q / float(i)),
                                                                                                          context.portfolio_value_memory[-1],
                                                                                                                           context.epsilon))

        _ = context.saver.save(context.sess, context.model_path)

def create_model(context):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    context.sess = tf.InteractiveSession(config=config)

    context.actor = ActorNetwork(context.sess,
                                 [len(context.assets), context.n, len(context.features)],
                                 len(context.assets) + 1,
                                 context.actor_learning_rate,
                                 context.tau,
                                 context.minibatch_size)

    context.critic = CriticNetwork(context.sess,
                                   [len(context.assets), context.n, len(context.features)],
                                   len(context.assets) + 1,
                                   context.critic_learning_rate,
                                   context.tau,
                                   context.gamma,
                                   context.actor.get_num_trainable_vars())

    # Inicializar las variables de Tensorflow
    context.sess.run(tf.global_variables_initializer())

    context.saver = tf.train.Saver()

    # Inicializar los pesos de las redes objetivo
    context.actor.update_target_network()
    context.critic.update_target_network()


def get_reward(context, asset_prices, last_asset_prices, portfolio_w):
    last_pv = context.portfolio_value_memory[-1]
    this_pv = np.sum(context.assets_quantity_invested * asset_prices) + context.train_cash
    context.assets_quantity_invested = (portfolio_w[1:] * this_pv) / asset_prices
    context.train_cash = portfolio_w[0] * this_pv
    reward = (this_pv / context.init_train_portfolio) - (last_pv / context.init_train_portfolio)
    context.portfolio_value_memory.append(this_pv)
    context.portfolio_w_memory.append(portfolio_w)
    """
    print("Last pv:", last_pv,
          "This pv:", this_pv,
          "Omega sum:", np.sum(portfolio_w),
          "Cash:", context.train_cash)
    """
    return reward


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
        end=pd.to_datetime('2018-3-1', utc=True),
    )
