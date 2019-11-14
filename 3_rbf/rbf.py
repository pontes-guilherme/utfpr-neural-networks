import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from clustering import kmeans
from functions import radial_basis_function

import random
import math


class RBFNetwork(object):

    def __init__(self, n_neurons, eta=0.01, method='lms'):
        self.n_neurons = n_neurons
        self.centers = None
        self.data_centers = None
        self.eta = eta
        self.weights = [random.randint(-1, 1) for _ in range(n_neurons)]
        self.method = method

    def rbf_function(self, center, data_point, sigma):
        return np.exp(-sigma*np.linalg.norm(center-data_point)**2)

    def _rbf_activation_matrix(self, X):
        matrix = np.zeros((len(X), self.n_neurons))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                sigma = self.sigmas[center_arg]
                matrix[data_point_arg, center_arg] = self.rbf_function(
                    center, data_point, sigma)
        return matrix

    def cluster(self, X):
        centers, data_centers, sigmas = kmeans(
            X, self.n_neurons, 'kmeanspp', 100)
        return centers, data_centers, sigmas

    def fit(self, X, Y):
        self.centers, self.data_centers, self.sigmas = self.cluster(X)

        if self.method == 'lms':
            matrix = self._rbf_activation_matrix(X)
            self.weights = np.dot(np.linalg.pinv(matrix), Y)

            return

        epochs = 1000
        for _ in range(epochs):
            for x, y_d in zip(X, Y):
                activations = []

                for center_data_belongs in range(len(self.centers)):
                    center = self.centers[center_data_belongs]
                    sigma = self.sigmas[center_data_belongs]

                    activations.append(self.rbf_function(center, x, sigma))

                y_pred_list = [w*a for w,
                               a in zip(self.weights, activations)]

                y_pred = sum(y_pred_list)

                error = y_d - y_pred

                self.weights = [w + self.eta*error*x_i for x_i,
                                w in zip(activations, self.weights)]

    def predict(self, X):
        if self.method == 'lms':
            matrix = self._rbf_activation_matrix(X)
            predictions = np.dot(matrix, self.weights)

            return predictions

        y_pred = []

        for x in X:
            activations = []

            for center_data_belongs in range(len(self.centers)):
                center = self.centers[center_data_belongs]
                sigma = self.sigmas[center_data_belongs]

                activations.append(self.rbf_function(center, x, sigma))

            y_pred_list = [w*a for w,
                           a in zip(self.weights, activations)]

            y_pred.append(sum(y_pred_list))

        return y_pred


iris = datasets.load_iris()

x = np.array(iris['data'])
y = np.array(iris['target'])

model = RBFNetwork(n_neurons=50, eta=0.01, method='lms')
model.fit(x, y)

y_pred = model.predict(x)

plt.plot(x, y, 'x', label='iris')
plt.plot(x, y_pred, 'o', label='RBF-Net')

plt.legend()
plt.show()
