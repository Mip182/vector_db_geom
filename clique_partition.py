#!/Users/mip182/opt/anaconda3/bin/python3

import numpy as np
import networkx as nx
from itertools import combinations, chain
from sklearn.neighbors import NearestNeighbors


class SimplexBuilder:
    def __init__(self, embeddings, k, t, metric='euclidean'):
        """
        Initializes the SimplexBuilder with embeddings, simplex size, number of neighbors, and metric.
        :param embeddings: np.array - An array of embeddings.
        :param k: int - The maximum size of the simplices (cliques) to find.
        :param t: int - The number of nearest neighbors to consider for building the graph.
        :param metric: str - The metric to use for distance calculation, default is 'euclidean'.
        """
        self.embeddings = embeddings
        self.k = k
        self.t = t
        self.metric = metric
        self.graph = nx.Graph()

    def build_knn_graph(self):
        """
        Builds a KNN graph using sklearn's NearestNeighbors, then symmetrizes the adjacency matrix.
        """
        knn = NearestNeighbors(n_neighbors=self.t, metric=self.metric)
        knn.fit(self.embeddings)
        distances, indices = knn.kneighbors(self.embeddings)

        n_samples = self.embeddings.shape[0]
        adj_matrix = np.zeros((n_samples, n_samples), dtype=int)

        for i, neighbors in enumerate(indices):
            for neighbor in neighbors:
                adj_matrix[i, neighbor] = 1

        sym_adj_matrix = adj_matrix + adj_matrix.T
        sym_adj_matrix[sym_adj_matrix > 1] = 1

        self.graph.add_nodes_from(range(n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if sym_adj_matrix[i, j] == 1:
                    self.graph.add_edge(i, j)

    def find_cliques(self):
        """
        Finds cliques of size up to k using the Bron-Kerbosch algorithm provided by networkx find_cliques.
        Adds all subsets of size <= k from each clique to the cliques_of_partition and sorts them by size.
        """
        cliques_of_partition = []
        all_cliques = list(nx.find_cliques(self.graph))

        for clique in all_cliques:
            for r in range(1, min(len(clique), self.k) + 1):
                cliques_of_partition.extend(list(combinations(clique, r)))

        cliques_of_partition = list(set(map(tuple, cliques_of_partition)))
        cliques_of_partition.sort(key=len)

        return cliques_of_partition

    def build_simplexes(self):
        """
        Builds real simplexes from the cliques.
        Converts each clique into a simplex by taking the corresponding embedding points.
        :return: List of np.array - Each array represents a simplex with its vertices.
        """
        cliques = self.find_cliques()
        simplexes = []

        for clique in cliques:
            simplex = np.array([self.embeddings[i] for i in clique])
            simplexes.append(simplex)

        return simplexes


if __name__ == "__main__":
    # Example usage
    embeddings = np.random.rand(10, 5)
    simplex_builder = SimplexBuilder(embeddings, k=3, t=4)
    simplex_builder.build_knn_graph()
    cliques = simplex_builder.find_cliques()
    print("Found cliques of size <= k:", cliques)
