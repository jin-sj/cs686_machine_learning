import os
import random
import numpy as np
import pandas as pd
from scipy.spatial import distance

MAX_ITERATIONS = 100
DISTANCE_EPSILON = 0.01

class KMeans:
    def __init__(self, k, data):
        self.k = k
        self.data = data
        self.clusters = [] * k
        self.centroids = [] * k

    def k_means_cluster(self):
        size = self.data.shape[0]
        initial_centroids_index = random.sample(range(size), self.k)

        _centroids = list(self.data[initial_centroids_index])
        _clusters = [[] for i in range(self.k)]

        iterations = 0
        converged = False
        while iterations < MAX_ITERATIONS or not converged:
            if iterations % 10 == 0:
                print("Currently at iteration: %s" % iterations)
            temp_clusters = [[] for i in range(self.k)]
            temp_centroids = [[] for i in range(self.k)]
            for i, row in enumerate(self.data):
                distances = [distance.euclidean(row, centroid) for centroid in _centroids]
                closest_centroid = np.argmin(distances)
                #temp_clusters[closest_centroid).append(self.data[i])
                #import pdb; pdb.set_trace()
                temp_clusters[closest_centroid].append(i)

            distance_moved = 0
            #for i in range(_centroids.shape[0]):
            for i in range(len(_centroids)):
                indices_of_cluster = temp_clusters[i]
                temp_centroids[i] = np.mean(self.data[indices_of_cluster], axis=0)
                distance_moved += distance.euclidean(_centroids[i], temp_centroids[i])

            _centroids = temp_centroids
            _clusters = temp_clusters

            if distance_moved > DISTANCE_EPSILON:
                converged = True
            iterations += 1

        self.centroids = _centroids
        self.clusters = _clusters

    def get_clusters(self):
        return (self.centroids, self.clusters)

def main():
    path = os.path.join(os.getcwd(), "data/country_gdp_growth_life_expectancy.csv")
    df = pd.read_csv(path)
    df = df[["GDP Growth", "Life Expectancy"]]
    df = df.dropna()

    k_means = KMeans(3, df.values)
    k_means.k_means_cluster()
    centroids, clusters = k_means.get_clusters()
    print(centroids)
    print(clusters)

if __name__ == "__main__":
    main()
