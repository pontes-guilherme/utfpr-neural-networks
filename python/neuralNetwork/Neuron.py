import random


class Neuron(object):

    _weights = []

    def __init__(self, eta=0.1, epochs=100, uses_batch=False,
                 batch_error_threshold=0.01):
        self._eta = eta
        self._epochs = epochs
        self._bias = 1
        self._uses_batch = uses_batch
        self._batch_error_threshold = batch_error_threshold

    def get_uses_batch(self):
        return self._uses_batch

    def get_batch_error_threshold(self):
        return self._batch_error_threshold

    def get_weights(self):
        return self._weights

    def get_epochs(self):
        return self._epochs

    def set_epochs(self, epochs):
        self._epochs = epochs

    def get_eta(self):
        return self._eta

    def set_eta(self, eta):
        self._eta = eta

    def get_bias(self):
        return self._bias

    def set_bias(self, bias):
        self._bias = bias

    def init_weights(self, size):
        self._weights = [random.uniform(-1, 1) for _ in range(size)]
        self.set_bias(1)

    def update_weights_sample(self, error, x_row):
        self._weights = [w + self.get_eta()*error*x_i for x_i,
                         w in zip(x_row, self.get_weights())]

    def update_weights_batch(self, delta_w):
        self._weights = [w + delta_w for w in self.get_weights()]

    def train(self, X, y):
        print("\nStarting neuron training...")

        input_size = len(X[0])
        self.init_weights(input_size)

        for epoch in range(self.get_epochs()):
            sum_squared_error = 0
            sum_error_batch = 0

            for x_row, y_d in zip(X, y):
                y_pred = self.predict(x_row)

                error = y_d - y_pred
                sum_squared_error += (error**2)

                self.set_bias(self.get_bias() + error*self.get_eta())

                if not self.get_uses_batch():
                    self.update_weights_sample(error, x_row)
                else:
                    for x_i in x_row:
                        sum_error_batch += (self.get_eta()*error*x_i)

            print('Epoch #%d' % epoch)

            mse = sum_squared_error/len(x_row)
            delta_w = sum_error_batch/len(x_row)

            error_threshold = self.get_batch_error_threshold() if self.get_uses_batch() else 0

            if self.should_stop(sum_squared_error, threshold=error_threshold):
                return

            if self.get_uses_batch():
                self.update_weights_batch(delta_w)

    def predict(self, x_row):
        prediction = sum(
            [x*w for x, w in zip(x_row, self.get_weights())]) + self.get_bias()
        return self.activate(prediction)

    def activate(self, prediction):
        return 1 if prediction > 0 else 0

    def should_stop(self, error, threshold=0):
        return error <= threshold


class AdalineNeuron(Neuron):
    def activate(self, prediction):
        return prediction
