"""
Implemetación del algoritmo DDGP (Deep Neural Network Policy Gradient)
para el caso de un sistema de trading sobre un solo activo
"""
import tensorflow as tf
import numpy as np
import tflearn

from buffer import ReplayBuffer

class ActorNetwork(object):
    """
    La entrada a red es el estado
    La salida es la acción bajo una política determinística mu(s)
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        self.initializer = tflearn.initializations.variance_scaling(factor=1.0,
                                                                    mode='FAN_IN',
                                                                    uniform=True,
                                                                    seed=None,
                                                                    dtype=tf.float32)

        # Red Actuadora
        self.inputs, self.softmax = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Red objetivo
        self.target_inputs, self.target_softmax = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Operación para actualizar periodicamente la red objetivo con los pesos de la red en línea
        self.update_target_network_params = [
            self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                 tf.multiply(self.target_network_params[i], 1.0 - self.tau))
            for i in range(len(self.target_network_params))]

        # Este gradiente es provisto por lar red crítica
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combinar los gradientes
        self.unnormalized_actor_gradients = tf.gradients(self.softmax, self.network_params, -self.action_gradient)
        grads = []
        for var, grad in zip(self.network_params, self.unnormalized_actor_gradients):
            if grad is None:
                grads.append(tf.zeros_like(var))
            else:
                grads.append(grad)
        self.unnormalized_actor_gradients = grads

        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size),
                                        self.unnormalized_actor_gradients))

        # Operación de optimización
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients,
                                                                                       self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)


    def create_actor_network(self):
        inputs = tf.placeholder(tf.float32, [None, self.s_dim[0], self.s_dim[1], self.s_dim[2]])

        net = tflearn.layers.conv.conv_2d(incoming=inputs,
                                          nb_filter=2,
                                          filter_size=[1, 3],
                                          strides=1,
                                          padding="valid",
                                          activation="relu",
                                          weights_init= self.initializer,
                                          weight_decay=0.0)

        width = net.get_shape()[2]

        net = tflearn.layers.conv.conv_2d(incoming=net,
                                          nb_filter=10,
                                          filter_size=[1, width],
                                          strides=1,
                                          padding="valid",
                                          activation="relu",
                                          weights_init=self.initializer,
                                          regularizer="L2",
                                          weight_decay=5e-09)

        net = tflearn.layers.conv.conv_2d(incoming=net,
                                          nb_filter=self.a_dim,
                                          filter_size=1,
                                          padding="valid",
                                          weights_init=self.initializer,
                                          regularizer="L2",
                                          weight_decay=5e-08)

        net = tflearn.fully_connected(net, self.a_dim, weights_init=self.initializer)

        softmax = tf.nn.softmax(net)

        return inputs, softmax

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={self.inputs: inputs,
                                                self.action_gradient: a_gradient})

    def predict(self, inputs):
        return self.sess.run(self.softmax, feed_dict={self.inputs: inputs})

    def predict_target(self, inputs):
        return self.sess.run(self.target_softmax, feed_dict={self.target_inputs: inputs})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

class CriticNetwork(object):
    """
    La entrada a la red viene dada por el estado y la acción.
    La salida es valor de función valor Q(s,a)
    La acción debe ser obtenida desde la salida de la red Actuadora
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        self.initializer = tflearn.initializations.variance_scaling(factor=1.0,
                                                                    mode='FAN_IN',
                                                                    uniform=True,
                                                                    seed=None,
                                                                    dtype=tf.float32)

        # Crear la red crítica
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Red Objetivo
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Operación para actualizar periódicamente las redes objetivo los pesos de la red en línea
        self.update_target_network_params = [
            self.target_network_params[i].assign(tf.multiply(self.target_network_params[i], self.tau) +
                                                 tf.multiply(self.target_network_params[i], 1.0 - self.tau))
            for i in range(len(self.target_network_params))]

        # Objetivo de la red (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Operación de optmización con pérdida
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Optener el gradiente de la red con respecto a la acción. Para cada acción en el minibatch esto sumará
        # los gradientes de cada salida de la red crítrica en el minibatch con respecto dicha acción. Cada salida
        # es independiente de todas las acciones excepto por una
        self.action_grads = tf.gradients(self.out, self.action)



    def create_critic_network(self):
        inputs = tf.placeholder(tf.float32, [None, self.s_dim[0], self.s_dim[1], self.s_dim[2]])
        action = tf.placeholder(tf.float32, [None, self.a_dim])

        net = tflearn.layers.conv.conv_2d(incoming=inputs,
                                          nb_filter=2,
                                          filter_size=[1, 3],
                                          strides=1,
                                          padding="valid",
                                          activation="relu",
                                          weights_init=self.initializer,
                                          weight_decay=0.0)

        width = net.get_shape()[2]

        net = tflearn.layers.conv.conv_2d(incoming=net,
                                          nb_filter=10,
                                          filter_size=[1, width],
                                          strides=1,
                                          padding="valid",
                                          activation="relu",
                                          weights_init=self.initializer,
                                          regularizer="L2",
                                          weight_decay=5e-09)

        net = tflearn.layers.conv.conv_2d(incoming=net,
                                          nb_filter=1,
                                          filter_size=1,
                                          padding="valid",
                                          weights_init=self.initializer,
                                          regularizer="L2",
                                          weight_decay=5e-08)

        net = tflearn.fully_connected(net, 20, weights_init=self.initializer,)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Agregar el tensor de acción
        # Se utilizan dos capas temporales para obtener los pasos y sesgos correspondientes
        t1 = tflearn.fully_connected(net, 10, weights_init=self.initializer,)
        t2 = tflearn.fully_connected(action, 10, weights_init=self.initializer,)

        net = tflearn.activation(tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation="relu")

        out = tflearn.fully_connected(net, 1, weights_init=self.initializer,)

        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.loss, self.optimize],
                             feed_dict={self.inputs: inputs,
                                        self.action: action,
                                        self.predicted_q_value: predicted_q_value})

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={self.inputs: inputs,
                                                  self.action: action})

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={self.target_inputs: inputs,
                                                         self.target_action: action})

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={self.inputs: inputs,
                                                           self.action: actions})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
