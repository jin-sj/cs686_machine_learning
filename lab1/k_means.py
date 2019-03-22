import os
import random
import numpy as np
import pandas as pd
from scipy.spatial import distance
from cluster import cluster
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics.cluster import v_measure_score

DISTANCE_EPSILON = 0.01

"""Runs k-means clustering algorithm"""
class KMeans(cluster):
    """Constructor
    :param k: Number of clusters
    :param max_iterations: Max iterations for the clustering
    """
    def __init__(self, k, max_iterations):
        super().__init__()
        self.k = k
        self.clusters = [[] for i in range(self.k)]
        self.centroids = [[] for i in range(self.k)]
        self.max_iterations = max_iterations
        self.size = 0

    """Fits given data to k-clusters through k-means
    :param X: 2-D data
    """
    def fit(self, X):
        super().fit(X)
        size = len(X)
        self.size = size
        initial_centroids_index = random.sample(range(size), self.k)

        _centroids = list(X[initial_centroids_index])
        _clusters = [[] for i in range(self.k)]

        iterations = 0
        converged = False
        while not converged:
            if iterations == self.max_iterations:
                break
            if iterations % 10 == 0:
                print("Currently at iteration: %s" % iterations)
            temp_clusters = [[] for i in range(self.k)]
            temp_centroids = [[] for i in range(self.k)]
            for i, row in enumerate(X):
                distances = [distance.euclidean(row, centroid) for centroid in _centroids]
                closest_centroid = np.argmin(distances)
                temp_clusters[closest_centroid].append(i)
            distance_moved = 0.
            for i in range(len(_centroids)):
                indices_of_cluster = temp_clusters[i]
                temp_centroids[i] = np.mean(X[indices_of_cluster], axis=0)
                distance_moved += distance.euclidean(_centroids[i], temp_centroids[i])

            _centroids = temp_centroids
            _clusters = temp_clusters
            print("distance moved: %.2f" % distance_moved)
            if distance_moved < DISTANCE_EPSILON:
                converged = True
            iterations += 1

        self.centroids = _centroids
        self.clusters = _clusters

    """Gets the cluster information"""
    def get_clusters(self):
        cluster_assignments = np.zeros(self.size, dtype=int)
        for i, cluster in enumerate(self.clusters):
            for point in cluster:
                cluster_assignments[point] = i
        return (self.centroids, self.clusters, cluster_assignments)

def main():
    k = 4
    max_iterations = 100
    X, cluster_assignments = make_blobs(n_samples=200, centers=k,
                                        cluster_std=0.60, random_state=0)
    k_means = KMeans(k, max_iterations)
    k_means.fit(X)
    centroids, clusters, cluster_assignments2 = k_means.get_clusters()
    score = v_measure_score(cluster_assignments, cluster_assignments2)
    print("score %.2f" % score)

if __name__ == "__main__":
    main()
