import pandas as pd
import networkx as nx
from itertools import combinations

# Create the dataframe
df = pd.DataFrame(
    {
        "paper_id": [1, 2, 3, 1, 1, 2, 3, 2, 2, 3, 3, 1],
        "author_id": ["A", "B", "C", "D", "E", "F", "G", "A", "D", "A", "D", "B"],
    }
)

# Create an empty graph
G = nx.Graph()

# For each paper, add edges between all pairs of authors
for paper_id in df["paper_id"].unique():
    authors = df[df["paper_id"] == paper_id]["author_id"].tolist()
    for pair in combinations(authors, 2):
        if G.has_edge(*pair):
            # if edge already exists, increment edge weight
            G[pair[0]][pair[1]]["weight"] += 1
        else:
            # else add new edge with weight=1
            G.add_edge(pair[0], pair[1], weight=1)

# Now, the number of collaborations between any pair of authors is just the weight of the edge between them
collaborations = {(u, v): G[u][v]["weight"] for u, v in G.edges()}

print(collaborations)
pass
