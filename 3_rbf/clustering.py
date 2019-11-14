import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
from sklearn.datasets.samples_generator import make_blobs


def kmeans(X, k, init, maxiter, seed=None):
    """
    specify the number of clusters k and
    the maximum iteration to run the algorithm
    """
    n_row, n_col = X.shape

    if seed is not None:
        np.random.seed(seed)

    if init == 'kmeanspp':
        # randomly choose the first centroid
        centroids = np.zeros((k, n_col))
        rand_index = np.random.choice(n_row)
        centroids[0] = X[rand_index]

        # compute distances from the first centroid chosen to all the other data points
        distances = pairwise_distances(
            X, [centroids[0]], metric='euclidean').flatten()

        for i in range(1, k):
            # choose the next centroid, the probability for each data point to be chosen
            # is directly proportional to its squared distance from the nearest centroid
            prob = distances ** 2
            rand_index = np.random.choice(n_row, size=1, p=prob / np.sum(prob))
            centroids[i] = X[rand_index]

            if i == k - 1:
                break

            # if we still need another cluster,
            # compute distances from the centroids to all data points
            # and update the squared distance as the minimum distance to all centroid
            distances_new = pairwise_distances(
                X, [centroids[i]], metric='euclidean').flatten()
            distances = np.min(np.vstack((distances, distances_new)), axis=0)

    else:  # random
        rand_indices = np.random.choice(n_row, size=k)
        centroids = X[rand_indices]

    for itr in range(maxiter):
        # compute distances between each data point and the set of centroids
        # and assign each data point to the closest centroid
        distances_to_centroids = pairwise_distances(
            X, centroids, metric='euclidean')
        cluster_assignment = np.argmin(distances_to_centroids, axis=1)

        # select all data points that belong to cluster i and compute
        # the mean of these data points (each feature individually)
        # this will be our new cluster centroids
        new_centroids = np.array(
            [X[cluster_assignment == i].mean(axis=0) for i in range(k)])

        # if the updated centroid is still the same,
        # then the algorithm converged
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    heterogeneity = 0
    sigmas = []
    for i in range(k):
        # note that pairwise_distance only accepts 2d-array
        cluster_data = X[cluster_assignment == i]
        distances = pairwise_distances(
            cluster_data, [centroids[i]], metric='euclidean')
        heterogeneity += np.sum(distances ** 2)

        sigmas.append(heterogeneity)

    return centroids, cluster_assignment, sigmas
