from dataclasses import dataclass
from typing import List
import numpy as np
import numpy.typing
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


################################ ERROS ################################
# 1. Por algum motivo o best_means está com o valor do current_means da última iteração. (CORRIGIDO -> precisava usar .copy() no current_means)
# 2. Por algum motivo o ponto (0, 0) está sendo adicionado e plotado no vetor de pontos depois de calcular os splits. (CORRIGIDO -> necessário troca a função split_arrays)
################################ FIM ERROS ################################

################################ TO-DO ##############################
# 1. 
################################ END ################################

@dataclass
class Split:
    split_point: numpy.typing.ArrayLike
    cost: float
    dim_split: int
    means: numpy.typing.ArrayLike

class Mean_Split:

    def __init__(self, data: np.ndarray, dims: int, n_clusters: int):
        self.data = data
        self.n_dims = dims
        self.n_clusters = n_clusters
        self.splits : List[Split] = []
        self.splits_arrays = []

    # create function that calculates the sum of the euclidian distance of a point from all points in a vector
    def sum_of_distances(self, point, points):
        return np.sum(np.linalg.norm(point - points, axis=1))

    def sum_of_squared_distances(self, point, points):
        return np.sum(np.square(point - points))

    def find_all_splits(self):
        split_obj, best_sorted_data = self.find_optimal_split(self.data)
        self.splits.append(split_obj)
        left_array, right_array = self.split_array(best_sorted_data, split_obj.split_point, split_obj.dim_split)
        self.splits_arrays.append(left_array)
        self.splits_arrays.append(right_array)
        for j in range(1, self.n_clusters - 1):
            best_cost = np.inf
            best_sorted_data = None
            best_split = None
            best_split_index = 0
            for i, split in enumerate(self.splits_arrays):
                split_obj, sorted_data = self.find_optimal_split(split)
                if split_obj.cost < best_cost:
                    best_cost = split_obj.cost
                    best_sorted_data = sorted_data
                    best_split = split_obj
                    best_split_index = i
            self.splits_arrays.pop(best_split_index)
            left_array, right_array = self.split_array(best_sorted_data, best_split.split_point, best_split.dim_split)
            self.splits_arrays.append(left_array)
            self.splits_arrays.append(right_array)
            self.splits.append(best_split)
        print(self.splits)

    def find_optimal_split(self, data: numpy.typing.ArrayLike):
        n_dims = data.shape[1]
        size = np.size(data, 0) - 1
        best_cost = np.inf
        split_point = [0, 0]
        dim_split = 0
        best_means = [0, 0]
        flag_dim = [0 for i in range(n_dims)]
        best_sorted_data = None
        for dim in range(n_dims):
            sorted_data = data[data[:, dim].argsort()]
            current_means = [sorted_data[0], np.mean(sorted_data[1:], axis=0)]
            aux = self.sum_of_distances(current_means[1], sorted_data[1:])
            if best_cost > aux:
                best_cost = aux
                split_point = sorted_data[0]
                dim_split = dim
                best_means = current_means
                best_sorted_data = sorted_data.copy()
                flag_dim[dim] = 1
            for i in range(1, size):
                current_means[0] = ((i*current_means[0]) + sorted_data[i])/(i+1)
                current_means[1] = (((size - i + 1) * current_means[1]) - sorted_data[i])/(size - i)
                current_cost = self.sum_of_distances(current_means[0], sorted_data[0:i+1]) + self.sum_of_distances(current_means[1], sorted_data[i+1:])
                #print(f"Split n: {i}, current cost = {current_cost}, current means = {current_means}")
                if current_cost < best_cost:
                    best_cost = current_cost
                    split_point = sorted_data[i]
                    dim_split = dim
                    best_means = current_means.copy()
                    if flag_dim[dim] == 0:
                        best_sorted_data = sorted_data.copy()
                        flag_dim[dim] = 1
        
        return Split(split_point, best_cost, dim_split, best_means), best_sorted_data

    #split array in two arrays, one with elements smaller than split_point and one 
    # with elements bigger than split_point on given dimension. the arrays must contain
    # the same elements present in the original array
    def split_array(self, data: numpy.typing.ArrayLike, split_point: numpy.typing.ArrayLike, dim: int):
        left_array = []
        right_array = []
        for i in range(np.size(data, 0)):
            if data[i, dim] <= split_point[dim]:
                left_array.append(data[i])
            else:
                right_array.append(data[i])
        return np.array(left_array), np.array(right_array)

    def plot_optimal_split(self):
        colors_tuple_list = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (0.5, 0.5, 0.5)]
        consolidated_data = np.concatenate(self.splits_arrays, axis=0)
        plt.scatter(consolidated_data[:, 0], consolidated_data[:, 1], s=10, color='black')
        for i in range(len(self.splits)):
            plt.scatter(self.splits[i].split_point[0], self.splits[i].split_point[1], s=50, color=colors_tuple_list[i])
            if self.splits[i].dim_split == 0:
                # plot vertical line crossing split point
                plt.axvline(x=self.splits[i].split_point[0], color=colors_tuple_list[i], linestyle='-')
            else:
                # plot horizontal line crossing split point
                plt.axhline(y=self.splits[i].split_point[1], color=colors_tuple_list[i], linestyle='-')
            plt.scatter(self.splits[i].means[0][0], self.splits[i].means[0][1], s=50, color=colors_tuple_list[i])
            plt.scatter(self.splits[i].means[1][0], self.splits[i].means[1][1], s=50, color=colors_tuple_list[i])
        plt.show()

# function that creates test datasets for optimal split and plots them with the optimal split
def create_optimal_split_data():
    data_list = []
    data = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0, random_state=0)[0]
    data_vertical = make_blobs(n_samples=100, n_features=2, centers=[[0, 10], [0, -10]], cluster_std=1.0, random_state=0)[0]
    data_horizontal = make_blobs(n_samples=100, n_features=2, centers=[[-10, 0], [10, 0]], cluster_std=1.0, random_state=0)[0]
    data_list.append(data)
    data_list.append(data_vertical)
    data_list.append(data_horizontal)
    return data_list

if __name__ == "__main__":
    data_list = create_optimal_split_data()
    teste = Mean_Split(data=data_list[0], dims=2, n_clusters=3)
    teste.find_all_splits()
    teste.plot_optimal_split()