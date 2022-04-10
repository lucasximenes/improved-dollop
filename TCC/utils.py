import numpy as np
import numpy.typing
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# create function that calculates the sum of the euclidian distance of a point from all points in a vector
def sum_of_distances(point, points):
    return np.sum(np.linalg.norm(point - points, axis=1))


def find_optimal_split(data: numpy.typing.ArrayLike):
    n_dims = data.shape[1]
    size = np.size(data, 0) - 1
    best_cost = np.inf
    split_point = [0, 0]
    dim_split = 0
    for dim in range(n_dims):
        sorted_data = data[data[:, dim].argsort()]
        current_means = [sorted_data[0], np.mean(sorted_data[1:], axis=0)]
        aux = sum_of_distances(current_means[1], sorted_data[1:])
        if best_cost > aux:
            best_cost = aux
            split_point = sorted_data[0]
            dim_split = dim
        for i in range(1, size):
            current_means[0] = (i*current_means[0] + sorted_data[i])/(i+1)
            current_means[1] = ((size- i + 1) * current_means[1] - sorted_data[i])/(size - i)
            current_cost = sum_of_distances(current_means[0], sorted_data[0:i+1]) + sum_of_distances(current_means[1], sorted_data[i+1:])
            print(f"Split n: {i}, current cost = {current_cost}")
            if current_cost < best_cost:
                best_cost = current_cost
                split_point = sorted_data[i]
                dim_split = dim
    return split_point, best_cost, dim_split

if __name__ == "__main__":
    #X, y = make_blobs(n_samples=101, centers=[[-5, 0], [5, 0]], n_features=2, random_state=0)
    X, y = make_blobs(n_samples=101, centers=[[0, 5], [0, -5]], n_features=2, random_state=0)
    # X, y = make_blobs(n_samples=101, centers=2 , n_features=2, random_state=0)
    x, y, z = find_optimal_split(X)
    print(x, y, z)
    plt.scatter(X[:, 0], X[:, 1])
    if z == 0:
        # plot vertical line crossing split point
        plt.axvline(x=x[0], color='r', linestyle='-')
    else:
        # plot horizontal line crossing split point
        plt.axhline(y=x[1], color='r', linestyle='-')
    plt.show()
