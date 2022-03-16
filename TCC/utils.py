import numpy as np
import numpy.typing
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=0)

def calculate_cost(center, points):
    dists = lambda x: np.linalg.norm(x - center)
    return np.sum(dists(points))

def find_optimal_split(data: numpy.typing.ArrayLike):
    X_sorted = data[data[:, 0].argsort()]
    Y_sorted = data[data[:, 1].argsort()]
    size = np.size(data, 0) - 1
    best_split = 0
    current_means = [X_sorted[0], np.mean(X_sorted[1:], axis=0)]
    best_cost = calculate_cost(current_means[1], X_sorted[1:])
    print(f"Initial best cost = {best_cost}")
    for i in range(1, size):
        current_means[0] = (i*current_means[0] + X_sorted[i])/(i+1)
        current_means[1] = ((size- i + 1) * current_means[1] - X_sorted[i])/(size - i)
        current_cost = calculate_cost(current_means[0], X_sorted[0:i+1]) + calculate_cost(current_means[1], X_sorted[i+1:])
        print(f"Split n: {i}, current cost = {current_cost}")
        if current_cost < best_cost:
            best_split = i
            best_cost = current_cost
    return X_sorted[best_split], best_cost, best_split


if __name__ == "__main__":
    x, y, z = find_optimal_split(X)
    print(x, y, z)
    plt.scatter(X[:, 0], X[:, 1])
    #plt.axhline(y=y, color='r', linestyle='-')
    plt.axvline(x=x[0], color='r', linestyle='-')
    plt.show()
