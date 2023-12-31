import numpy as np
from sklearn import cluster, datasets, mixture
import matplotlib.pyplot as plt

__all__ = ['simulated_dataset', 'mac_queen_initialisation', 'plot_fs']

def simulated_dataset():
    return datasets.make_blobs(n_samples=1000, cluster_std=[2.0, 2.5, 1.5], random_state=42)

def mac_queen_initialisation(X, k):
    centroids = np.zeros((k, X.shape[1]))
    for cc in range(k):
        index = np.random.randint(0, high=X.shape[0])
        centroids[cc] = X[index]
    return centroids


def plot_fs(predict_func, X, y, resolution = .01, save_name=None):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    Z = predict_func(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    if save_name is not None:
        plt.savefig(save_name)
    plt.show()

