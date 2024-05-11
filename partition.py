#!/Users/mip182/opt/anaconda3/bin/python3

import numpy as np
from clique_partition import SimplexBuilder


class EmbeddingsPartition:
    def __init__(self, embeddings, dimension):
        """
        Initializes the class with an array of embeddings and their dimension.
        :param embeddings: np.array - An array of embeddings.
        :param dimension: int - The dimension of each embedding.
        """
        self.embeddings = embeddings
        self.dimension = dimension

    def run_method(self, method_name, **kwargs):
        """
        Executes the specified embedding processing method with given parameters.
        :param method_name: str - The name of the method.
        :param kwargs: dict - The parameters for the method.
        """
        internal_method_name = method_name + "_method"
        if hasattr(self, internal_method_name):
            method = getattr(self, internal_method_name)
            return method(**kwargs)
        else:
            raise ValueError(f"Method {method_name} is not supported.")

    def clique_method(self, k, t, metric='euclidean'):
        """
        Uses the SimplexBuilder to build a graph with embeddings and find cliques of size k.
        :param k: int - Size of the cliques to find.
        :param t: int - The number of nearest neighbors to consider for building the graph.
        :param metric: str - Metric used for distance calculation.
        :return: List of cliques found in the graph.
        """
        simplex_builder = SimplexBuilder(self.embeddings, k, t, metric)
        simplex_builder.build_knn_graph()
        cliques_of_partition = simplex_builder.find_cliques()
        return cliques_of_partition


if __name__ == "__main__":
    # Example of usage
    embeddings = np.random.rand(10, 5)  # 10 embeddings, each with 5 dimensions
    partitioner = EmbeddingsPartition(embeddings, 5)
    cliques = partitioner.run_method("clique", k=3, t=4)
    print("Found cliques:", cliques)
