import random
import numpy as np


class Sigmoid:
    @staticmethod
    def apply(p):
        return 1 / (1 + np.exp(p))

    @staticmethod
    def derivative(value):
        return value * (1 - value)


class Layer:
    def __init__(self, inputs_size, qtd_neurons, activation_function=None,
                 weights=None, layer_bias=None):
        self.__weights = weights if weights is not None else np.random.rand(
            inputs_size, qtd_neurons)
        self.__layer_bias = layer_bias if layer_bias is not None else np.random.rand(
            qtd_neurons)
        self.__activation_function = activation_function

        self.last_activation = None
        self.error = None
        self.delta = None

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, weights):
        self.__weights = weights

    @property
    def layer_bias(self):
        return self.__layer_bias

    @layer_bias.setter
    def layer_bias(self, layer_bias):
        self.__layer_bias = layer_bias

    @property
    def activation_function(self):
        return self.__activation_function

    def activate(self, x_row):
        layer_activation = np.dot(x_row, self.weights) + self.layer_bias
        self.activation_history = self.exec_activation_function(
            layer_activation)
        self.last_activation = self.activation_history

        return self.activation_history

    def exec_activation_function(self, prediction):
        """Executa a função de ativação em uma lista de 
        valores preditos

        Arguments:
            prediction {list} -- Lista de predições

        Returns:
            list -- Lista de valores de ativação
        """
        if self.activation_function == 'sigmoid':
            # print(Sigmoid.apply(prediction))
            return Sigmoid.apply(prediction)

        return prediction

    def exec_function_derivative(self, prediction):
        if self.activation_function == 'sigmoid':
            return Sigmoid.derivative(prediction)

        return prediction


class NN:

    __layers = []

    def __init__(self, eta=0.01, epochs=1000):
        self.__eta = eta
        self.__epochs = epochs

    @property
    def layers(self):
        return self.__layers

    @property
    def eta(self):
        return self.__eta

    @property
    def epochs(self):
        return self.__epochs

    def new_layer(self, layer_to_add):
        self.__layers.append(layer_to_add)

    def propagate(self, x_row):
        for l in self.layers:
            x_row = l.activate(x_row)

        return x_row

    def backpropagate(self, x_row, y):
        """Retropropaga os erros da última camada para as demais

        Arguments:
            x_row {list} -- Representa um padrão de treinamento
            y {float} -- Saída desejada
        """
        y_pred = self.propagate(x_row)

        layers_backwards = reversed(range(len(self.layers[:-1])))

        last_layer = self.layers[-1]
        last_layer.error = y - y_pred
        last_layer.delta = last_layer.error * \
            last_layer.exec_function_derivative(y_pred)

        for lidx in layers_backwards:
            layer = self.layers[lidx]
            next_layer = self.layers[lidx + 1]
            layer.error = np.dot(next_layer.weights, next_layer.delta)
            layer.delta = layer.error * \
                layer.exec_function_derivative(layer.last_activation)

        for lidx in range(len(self.layers)):
            layer = self.layers[lidx]
            # The input is either the previous layers output or X itself (for the first hidden layer)
            input_to_use = np.atleast_2d(self.layers[lidx - 1].last_activation)
            layer.weights += layer.delta * input_to_use.T * self.eta

        return last_layer.error

    def train(self, X, y):
        """Treina a rede neural

        Arguments:
            X {list} -- Lista de padrões de treinamento
            y {list} -- Lista de valores desejados, um para cada padrão
        """
        for epoch in range(self.epochs):
            sum_squared_error = 0

            for p in range(len(X)):
                errors_last_layer = self.backpropagate(X[p], y[p])

                sum_squared_error += np.sum(np.abs(errors_last_layer)) ** 2

            mse = sum_squared_error/len(X)
            print('epoch #', epoch, ' - mse:', mse)
            if mse == 0:
                return

    def predict(self, x_row):
        """Efetua uma predição para um padrão de entrada

        Arguments:
            x_row {list} -- Padrão de entrada

        Returns:
            [float] -- Valor predito pela rede
        """
        propagation = self.propagate(x_row)

        return np.argmax(propagation, axis=1)


nn = NN(0.01, 10000)
nn.new_layer(Layer(2, 3, 'sigmoid'))
nn.new_layer(Layer(3, 3, 'sigmoid'))
nn.new_layer(Layer(3, 2, 'sigmoid'))

# Define dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

# Train the neural network
errors = nn.train(X, y)
print(nn.predict(X))
