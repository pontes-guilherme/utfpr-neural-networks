import random
import numpy as np

class Neuron(object):

    __weights = []

    def __init__(self, eta=0.1, epochs=100, uses_batch=False,
                 error_threshold=0.01, validation_X=None, validation_y=None):
        self.__eta = eta
        self.__epochs = epochs
        self.__bias = 1
        self.__uses_batch = uses_batch
        self.__error_threshold = error_threshold
        self.__validation_X = validation_X
        self.__validation_y = validation_y

    @property
    def uses_batch(self):
        return self.__uses_batch

    @property
    def error_threshold(self):
        return self.__error_threshold

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, weights):
        self.__weights = weights

    @property
    def epochs(self):
        return self.__epochs

    @property
    def eta(self):
        return self.__eta

    @property
    def bias(self):
        return self.__bias

    @bias.setter
    def bias(self, bias):
        self.__bias = bias

    @property
    def validation_X(self):
        return self.__validation_X

    @property
    def validation_y(self):
        return self.__validation_y

    def init_weights(self, size):
        self.weights = np.random.random(size)
        #self.bias = 1

    def update_weights_sample(self, error, x_row):
        self.weights = np.add(
            self.weights, np.multiply(x_row, self.eta * error)
        )

    def update_weights_batch(self, delta_w):
        self.weights = np.add(self.weights, delta_w)

    def print_weights(self):
        print("\n")
        print('#'*50)
        print('list of weights - first one is the bias')
        print(self.weights)
        print('#'*50)

    def train(self, X, y):
        print("\nStarting neuron training...")

        X = np.array(X)

        bias_column = np.ones((X.shape[0], 1))
        X = np.hstack((bias_column, X))

        n_lines, n_columns = X.shape

        self.init_weights(n_columns)
        for epoch in range(self.epochs):
            sum_squared_error = 0
            sum_error_w_batch = np.zeros(n_columns)

            for x_row, y_d in zip(X, y):
                y_pred = self.predict(x_row)

                error = y_d - y_pred
                sum_squared_error += (error ** 2)

                # self.bias = self.bias + (error * self.eta)

                if not self.uses_batch:
                    self.update_weights_sample(error, x_row)
                else:
                    sum_error_w_batch = np.add(
                        sum_error_w_batch, np.multiply(x_row, self.eta * error)
                    )

            if self.validation_X:
                sum_squared_error = 0

                for x_validation, y_validation in zip(self.validation_X, self.validation_y):
                    y_valid_pred = self.predict(x_validation)

                    error_validation = y_validation - y_valid_pred
                    sum_squared_error += (error_validation ** 2)

            mse = sum_squared_error/len(x_row)
            delta_w = np.divide(sum_error_w_batch, len(x_row))

            print('Epoch #', epoch, " - mse: {:.10f}".format(mse))

            if self.should_stop(sum_squared_error,
                                threshold=self.error_threshold):
                self.print_weights()
                return

            if self.uses_batch:
                self.update_weights_batch(delta_w)

        self.print_weights()

    def predict(self, x_row):
        if len(x_row) != len(self.weights):
            x_row = [1] + x_row
        prediction = np.dot(self.weights, x_row)
        return self.activate(prediction)

    def activate(self, prediction):
        # return 1 if prediction > 0 else 0
        return prediction

    def should_stop(self, error, threshold=0):
        return error <= threshold