from .Neuron import AdalineNeuron


class Adaline(object):

    _neurons = []
    _labels = []

    def __init__(self, training_sets_list, results_list, labels=None, eta=0.1,
                 epochs=100, uses_batch=False, batch_error_threshold=0.01):

        self._neurons = []
        self._labels = labels if labels else []

        i = 0

        for X, y in zip(training_sets_list, results_list):
            n = AdalineNeuron(eta, epochs, uses_batch,
                              batch_error_threshold)

            n.train(X, y)
            self._neurons.append(n)

            if not labels:
                self._labels.append('Neuron %s:' % i)

            i += 1

    def predict(self, x_row):
        print("\nPrediction for %s" % x_row)

        for n, label in zip(self._neurons, self._labels):
            print(label, ": ", n.predict(x_row))
