# Standard scientific libraries:
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

# My own library:
from auxiliaries import *


def plot_centroids(X, ownership, centroids, k):
    for i in range(k):
        plt.plot(X[ownership == i, 0], X[ownership == i, 1], "o", label=f"Set {i + 1}")
        plt.plot(centroids[i][0], centroids[i][1], 'ok')
        plt.legend()
    return None


def k_means_clustering(X, k):
    # Compute the initial guess for the centroids:
    centroids = mac_queen_initialisation(X, k=k)
    iter = 0
    for i in range(100):  # 100 is just a magical number to guarantee non-infinite loop.
        iter += 1
        # Compute the distance between each point and each centroid:
        point2centroid_dist = cdist(X, centroids)

        # Which points are nearer to which centroids:
        belong2which_centroid = np.argmin(point2centroid_dist, axis=1)

        # Assign the points to its nearest centroid:
        new_centroids = np.array([
            np.mean(X[belong2which_centroid == i], axis=0)
            if True in [belong2which_centroid == i][0]
            else centroids[i]
            for i in range(k)
        ])

        # Compute the difference between the new centroids and the former:
        error = np.linalg.norm(new_centroids - centroids)

        # Update the new centroids:
        centroids = new_centroids

        if error < 1e-5:  # If the new centroids are too similar to the old, the algorithm has converged.
            print("Successful run")
            break
    return centroids, belong2which_centroid, iter


if __name__ == "__main__":
    # Generate the data:
    X, _ = simulated_dataset()

    # Select the number of clusters:
    k = 3

    # Compute the clusters centroids:
    centroids, ownership, max_iter = k_means_clustering(X, k)

    # Plot the result:
    plot_centroids(X, ownership, centroids, k)
