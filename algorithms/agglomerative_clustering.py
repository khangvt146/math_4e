import networkx as nx
import numpy as np
import community
import time
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from tqdm.auto import tqdm

DATASET_PATH = "data/facebook_combined.txt"
# DATASET_PATH = "data/test.txt"


class AggHierarchicalClustering:
    """
    Community Detection with Agglomerative Hierarchical Clustering Algorithms
    """

    def __init__(self) -> None:
        self.graph = None

    def load_dataset(self, file_path: str = DATASET_PATH):
        self.graph: nx.Graph = nx.read_edgelist(file_path)

    def _calculate_distance_matrix(self) -> np.array:
        num_nodes = len(self.graph.nodes)

        # Initialize the Distance Matrix
        distance_matrix = np.zeros((num_nodes, num_nodes))

        # Fill the distance matrix
        for i in range(num_nodes):
            for j in range(num_nodes):
                # Get the set of neighbors for nodes i and j
                neighbors_i = set(self.graph.neighbors(str(i)))
                neighbors_j = set(self.graph.neighbors(str(j)))
                neighbor_ij = neighbors_i.intersection(neighbors_j)

                # d_ij = k_i + k_j - 2*n_ij
                distance_matrix[i][j] = (
                    len(neighbors_i) + len(neighbors_j) - 2 * len(neighbor_ij)
                )
        return distance_matrix

    def run(
        self, distance_matrix: np.array, n_clusters: int = 10, linkage: str = "single"
    ):
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage=linkage
        )
        clustering.fit(distance_matrix)

        partition = {}
        for num, data in enumerate(list(clustering.labels_)):
            partition[str(num)] = data

        return partition

    def benchmark(self):
        # Initialize test medthod
        linkage_list = ["single", "complete", "average"]
        n_clusters = [i for i in range(1, 20 + 1)]

        single_modularity_lst = []
        single_execution_lst = []

        complete_modularity_lst = []
        complete_execution_lst = []

        average_modularity_lst = []
        average_execution_lst = []

        distance_matrix = self._calculate_distance_matrix()
        pbar = tqdm(total=len(n_clusters) * len(linkage_list))

        for linkage in linkage_list:
            for num in n_clusters:
                start_time = time.time()
                partition = self.run(distance_matrix, n_clusters=num, linkage=linkage)
                modularity = community.modularity(partition, self.graph)
                end_time = time.time()

                if linkage == "single":
                    single_modularity_lst.append(modularity)
                    single_execution_lst.append(end_time - start_time)

                elif linkage == "complete":
                    complete_modularity_lst.append(modularity)
                    complete_execution_lst.append(end_time - start_time)

                elif linkage == "average":
                    average_modularity_lst.append(modularity)
                    average_execution_lst.append(end_time - start_time)
                
                pbar.update(1)

        pbar.close()
        # Plot modularity score for each linkage function
        plt.figure()
        plt.plot(n_clusters, single_modularity_lst, label="single-linkage")
        plt.plot(n_clusters, complete_modularity_lst, label="complete-linkage")
        plt.plot(n_clusters, average_modularity_lst, label="average-linkage")
        plt.title("Modularity score with different linkage function")
        plt.xticks(n_clusters)
        plt.legend()
        plt.savefig("images/agg_clustering_modularity.png")

        # Plot execution time for each linkage function
        plt.figure()
        plt.plot(n_clusters, single_execution_lst, label="single-linkage")
        plt.plot(n_clusters, complete_execution_lst, label="complete-linkage")
        plt.plot(n_clusters, average_execution_lst, label="average-linkage")
        plt.title("Execution time with different linkage function")
        plt.xticks(n_clusters)
        plt.legend()
        plt.savefig("images/agg_clustering_execution_time.png")


if __name__ == "__main__":
    agg_cluster = AggHierarchicalClustering()
    agg_cluster.load_dataset()
    agg_cluster.benchmark()
