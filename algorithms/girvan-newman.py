import networkx as nx
import numpy as np
import community
import time
from tqdm.auto import tqdm

DATASET_PATH = "data/facebook_combined.txt"
# DATASET_PATH = "data/test.txt"

class GirvanNewman:
    """
    Community Detection with Girvan-Newman Algorithms   
    """
    def __init__(self) -> None:
        self.graph = None
    
    def load_dataset(self, file_path: str = DATASET_PATH):
        self.graph: nx.Graph = nx.read_edgelist(file_path)
    

    def run(self):
        modularities = []
        num_communities = []
        all_communities = []
    
        pbar = tqdm(total=self.graph.number_of_edges() - 1)

        start_time = time.time()
        while self.graph.number_of_edges() > 1:
            edge_betweenness = nx.edge_betweenness_centrality(self.graph)
            max_edge = max(edge_betweenness, key=edge_betweenness.get)
            self.graph.remove_edge(max_edge[0], max_edge[1])

            communities = list(nx.connected_components(self.graph))
            modularity_score =  nx.algorithms.community.quality.modularity(self.graph, communities)

            modularities.append(modularity_score)
            num_communities.append(len(communities))
            all_communities.append(communities)
            pbar.update(1)

        pbar.close()
        max_modularity_index = modularities.index(max(modularities))
        optimal_num_communities = num_communities[max_modularity_index]
        optimal_communities = all_communities[max_modularity_index]
        
        print(f"Number of communities: {optimal_num_communities} and Modularity: {max(modularities)}")
        print("Time Execution: ", time.time()- start_time)



if __name__ == "__main__":
    agg_cluster = GirvanNewman()
    agg_cluster.load_dataset()
    agg_cluster.run()