from .Neuron import Neuron


class Perceptron(object):

    _neurons = []
    _labels = []

    def __init__(self, training_sets_list, results_list, labels=None, eta=0.1,
                 epochs=100, uses_batch=False, error_threshold=0.01,
                 validatin_X=None, validation_y=None):

        self._neurons = []
        self._labels = labels if labels else []

        i = 0

        for X, y in zip(training_sets_list, results_list):
            n = Neuron(eta, epochs, uses_batch,
                       error_threshold, validatin_X, validation_y)

            n.train(X, y)
            self._neurons.append(n)

            if not labels:
                self._labels.append('Neuron %s' % i)

            i += 1

    def predict(self, x_row):
        print("\nPrediction for %s" % x_row)

        for n, label in zip(self._neurons, self._labels):
            print(label, ": ", n.predict(x_row))
