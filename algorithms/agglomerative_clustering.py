import networkx as nx
import numpy as np
import community
import time
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from tqdm.auto import tqdm

DATASET_PATH = "data/graph.txt"
# DATASET_PATH = "data/test.txt"


class AggHierarchicalClustering:
    """
    Community Detection with Agglomerative Hierarchical Clustering Algorithms
    """

    def __init__(self) -> None:
        self.graph = None

    def load_dataset(self, file_path: str = DATASET_PATH):
        self.graph: nx.Graph = nx.read_weighted_edgelist(file_path, nodetype=int)

    def _map_node_id(self):
        node_list = list(self.graph.nodes)
        node_list.sort()
        map_node_id = {i: int(value) for i, value in enumerate(node_list)}
        return map_node_id

    def _calculate_distance_matrix_unweighted(self) -> np.array:
        # Create dict to map node_id from 0 -> len(node_list)
        self.map_node_id = self._map_node_id()
        num_nodes = len(self.map_node_id)

        # Initialize the Distance Matrix
        distance_matrix = np.zeros((num_nodes, num_nodes))

        # Fill the distance matrix
        for i in self.map_node_id:
            for j in self.map_node_id:
                # Get the set of neighbors for nodes i and j
                neighbors_i = set(self.graph.neighbors(self.map_node_id[i]))
                neighbors_j = set(self.graph.neighbors(self.map_node_id[j]))
                neighbor_ij = neighbors_i.intersection(neighbors_j)

                # d_ij = k_i + k_j - 2*n_ij
                distance_matrix[i][j] = (
                    len(neighbors_i) + len(neighbors_j) - 2 * len(neighbor_ij)
                )
                if self.graph.has_edge(self.map_node_id[i], self.map_node_id[j]):
                    distance_matrix[i][j] = (
                        distance_matrix[i][j]
                        + self.graph[self.map_node_id[i]][self.map_node_id[j]]["weight"]
                    )
        return distance_matrix

    def _calculate_distance_matrix_weighted(self) -> np.array:
        # Create dict to map node_id from 0 -> len(node_list)
        self.map_node_id = self._map_node_id()
        num_nodes = len(self.map_node_id)

        # Initialize the Distance Matrix
        distance_matrix = np.zeros((num_nodes, num_nodes))

        # Fill the distance matrix
        for i in self.map_node_id:
            for j in self.map_node_id:
                distance_matrix[i][j] = self._calculate_weight_distance(self.map_node_id[i], self.map_node_id[j])
        return distance_matrix

    def _calculate_weight_distance(self, node_1: int, node_2: int):
        if self.graph.has_edge(node_1, node_2):
            s_node_1 = sum(weight for _, _, weight in self.graph.edges(node_1, data='weight'))
            s_node_2 = sum(weight for _, _, weight in self.graph.edges(node_2, data='weight'))
            w_node1_node2 = self.graph[node_1][node_2]['weight']
            distance = s_node_1 + s_node_2 - 2 * w_node1_node2
        else:
            distance = 99999

        return distance

    def run(
        self, distance_matrix: np.array, n_clusters: int = 10, linkage: str = "single"
    ):
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage=linkage
        )
        clustering.fit(distance_matrix)

        partition = {}
        for num, data in enumerate(list(clustering.labels_)):
            partition[self.map_node_id[num]] = data

        return partition

    def benchmark(self):
        # Initialize test medthod
        linkage_list = ["single", "complete", "average"]
        n_clusters = [i for i in range(1, 30 + 1)]

        single_modularity_lst = []
        single_execution_lst = []

        complete_modularity_lst = []
        complete_execution_lst = []

        average_modularity_lst = []
        average_execution_lst = []

        distance_matrix = self._calculate_distance_matrix_weighted()
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
        plt.figure(figsize=(12, 8))
        plt.plot(n_clusters, single_modularity_lst, label="single-linkage")
        plt.plot(n_clusters, complete_modularity_lst, label="complete-linkage")
        plt.plot(n_clusters, average_modularity_lst, label="average-linkage")
        plt.title("Modularity score with different linkage function")
        plt.xticks(n_clusters)
        plt.legend()
        plt.savefig("images/agg_clustering_modularity.png")
        plt.xlabel('Community')
        plt.ylabel('Modularity score')  

        # Plot execution time for each linkage function
        plt.figure(figsize=(12, 8))
        plt.plot(n_clusters, single_execution_lst, label="single-linkage")
        plt.plot(n_clusters, complete_execution_lst, label="complete-linkage")
        plt.plot(n_clusters, average_execution_lst, label="average-linkage")
        plt.title("Execution time with different linkage function")
        plt.xticks(n_clusters)
        plt.legend()
        plt.savefig("images/agg_clustering_execution_time.png")
        plt.xlabel('Community')
        plt.ylabel('Execution time (s)')


if __name__ == "__main__":
    agg_cluster = AggHierarchicalClustering()
    agg_cluster.load_dataset()
    agg_cluster.benchmark()
