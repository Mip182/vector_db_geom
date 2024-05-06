import numpy as np
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from itertools import combinations


class SimplexBuilder:
    def __init__(self, embeddings, k, eps, metric='euclidean'):
        """
        Initializes the SimplexBuilder with embeddings, simplex size, distance threshold, and metric.
        :param embeddings: np.array - An array of embeddings.
        :param k: int - The size of the simplices (cliques) to find.
        :param eps: float - The maximum distance between connected embeddings.
        :param metric: str - The metric to use for distance calculation, default is 'euclidean'.
        """
        self.embeddings = embeddings
        self.k = k
        self.eps = eps
        self.metric = metric
        self.graph = nx.Graph()

    def build_graph(self):
        """
        Builds a graph where each node represents an embedding and edges connect nodes
        that are within eps distance of each other according to the specified metric.
        """
        distances = squareform(pdist(self.embeddings, metric=self.metric))
        for i in range(len(distances)):
            for j in range(i + 1, len(distances)):
                if distances[i][j] <= self.eps:
                    self.graph.add_edge(i, j)

    def find_cliques(self):
        """
        Finds cliques of size k, iteratively removing vertices once they are part of a found clique or alone.
        """
        cliques_of_partition = []
        G = self.graph.copy()
        nodes = list(G.nodes())
        i = 0
        while i < len(nodes):
            found = False
            current_node = nodes[i]
            for clique in combinations(set(G.neighbors(current_node)) | {current_node}, self.k):
                if len(clique) == self.k and G.subgraph(clique).number_of_edges() == self.k * (self.k - 1) / 2:
                    cliques_of_partition.append(list(clique))
                    G.remove_nodes_from(clique)
                    nodes = [node for node in nodes if node not in clique]
                    found = True
                    break
            if not found:
                G.remove_node(current_node)
                nodes.remove(current_node)
            i = 0
        return cliques_of_partition


# Example usage
if __name__ == "__main__":
    embeddings = np.random.rand(10, 5)
    simplex_builder = SimplexBuilder(embeddings, k=3, eps=0.5)
    simplex_builder.build_graph()
    cliques = simplex_builder.find_cliques()
    print("Found cliques of size k:", cliques)
