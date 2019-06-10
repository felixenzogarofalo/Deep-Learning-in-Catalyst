# Este en un módulo con heramientas varias
import numpy as np
import math

# Crear Trajectorias de Monte Carlo
def mc_trajectories(data, N_b, T):
    """
    Crear N_b trajectorias de Monte Carlo con una longitud de T
    :param data: DataFrame de la forma (time_step x 1). Un solo asset
    :param N_b: Número de trajectorias
    :param T: Longitud de trajectorias
    :return: Array de Numpy de la forma (N_b x T)
    """

    # Calcular la media de los valores
    mu = data.mean()

    # Calcular la volatilidad de los valores a partir de los retornos poncentuales
    returns = data.pct_change()
    vol = returns.std()

    # Precios inciales
    s_price = data.iloc[-1]
    trajectories = []
    for i in range(N_b):
        # Crear una lista de retornos diarios utilizando un distribución aleatoria Normal
        generated_returns = np.random.normal(loc=mu/T, scale=vol/math.sqrt(T), size=T - 1) + 1

        price_list = [s_price]
        for step in generated_returns:
            price_list.append(price_list[-1] * step)
        trajectories.append(price_list)

    t_array = np.array(trajectories)

    return t_array