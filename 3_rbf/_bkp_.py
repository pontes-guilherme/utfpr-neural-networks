import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from clustering import kmeans
from functions import radial_basis_function


class RBFN(object):

    def __init__(self, n_neurons):
        """ radial basis function network
        # Arguments
            input_shape: dimension of the input data
            e.g. scalar functions have should have input_dimension = 1
            n_neurons: the number
            n_neurons: number of hidden radial basis functions,
            also, number of centers.
        """
        self.n_neurons = n_neurons
        self.centers = None
        self.weights = None

    def _kernel_function(self, center, data_point, sigma):
        return np.exp(-sigma*np.linalg.norm(center-data_point)**2)

    def _calculate_interpolation_matrix(self, X):
        """ Calculates interpolation matrix using a kernel_function
        # Arguments
            X: Training data
        # Input shape
            (num_data_samples, input_shape)
        # Returns
            G: Interpolation matrix
        """
        G = np.zeros((len(X), self.n_neurons))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                sigma = self.sigmas[center_arg]
                G[data_point_arg, center_arg] = self._kernel_function(
                        center, data_point, sigma)
        return G

    def _select_centers(self, X):
        # random_args = np.random.choice(len(X), self.n_neurons)
        # centers = X[random_args]
        centers, _, sigmas = kmeans(X, self.n_neurons, 'kmeanspp', 100)
        return centers, sigmas

    def fit(self, X, Y):
        """ Fits weights using linear regression
        # Arguments
            X: training samples
            Y: targets
        # Input shape
            X: (num_data_samples, input_shape)
            Y: (num_data_samples, input_shape)
        """
        self.centers, self.sigmas = self._select_centers(X)
        G = self._calculate_interpolation_matrix(X)
        self.weights = np.dot(np.linalg.pinv(G), Y)

    def predict(self, X):
        """
        # Arguments
            X: test data
        # Input shape
            (num_test_samples, input_shape)
        """
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)

        return predictions


iris = datasets.load_iris()

x = np.array(iris['data'])
y = np.array(iris['target'])

# fitting RBF-Network with data
model = RBFN(n_neurons=100)
model.fit(x, y)


y_pred = model.predict(x)

plt.plot(x, y, 'x', label='iris')
plt.plot(x, y_pred, 'o', label='RBF-Net')
plt.legend()

plt.tight_layout()
plt.show()
