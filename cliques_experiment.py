#!/Users/mip182/opt/anaconda3/bin/python3

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from partition import EmbeddingsPartition
from projections import SimplexProjection


# I want to conduct an experiment -
# I have dataset with embeddings,
# I want to split it by various part to train and test datasets (let's say 0.3, 0.5 and 0.7).
# Then I want to build simplexes onto train dataset with let's say k=3,k=4,k=5
# and eps on a grid from 0.0001 to 0.5 with some step,
# and then I want to use test dataset,
# to calculate what is the number of points in it lies in simplexes that we build, and also some statistics,
# like quantiles of distances between point and projection,
# and the histogram that have the number of points,
# which projection corresponds to exact simplex

def split_data(embeddings, ratios):
    splits = {}
    for ratio in ratios:
        train, test = train_test_split(embeddings, test_size=ratio, random_state=42)
        splits[ratio] = (train, test)
    return splits


def build_simplexes(train_data, k_values, eps_grid):
    simplexes = {}
    for k in k_values:
        for eps in eps_grid:
            partitioner = EmbeddingsPartition(train_data, train_data.shape[1])
            cliques = partitioner.run_method('clique', k=k, eps=eps)
            if len(cliques) > 0:
                simplexes[(k, eps)] = cliques
    return simplexes


def project_points(test_data, simplexes):
    results = []
    for (k, eps), cliques in simplexes.items():
        projector = SimplexProjection(cliques)
        if len(cliques) > 0:
            closest_projections, projection_indexes = projector.find_closest_projection(test_data)
            distances = np.linalg.norm(test_data - closest_projections, axis=1)
            inside_simplex = projector.is_projections_in_simplexes(test_data)
            results.append((k, eps, distances, projection_indexes, inside_simplex))
        else:
            results.append((k, eps, np, np.ones(test_data.shape[0]) * float('inf'), np.ones(test_data.shape[0]) * -1))
    return results


def analyze_results(results):
    plt.figure(figsize=(10, 6))
    for k, eps, distances, projection_indexes, inside_simplex in results:
        num_inside = np.sum(inside_simplex)
        label = f'k={k}, eps={eps:.4f}'
        print(f"k={k}, eps={eps}")
        print("Number inside simplexes:", num_inside)
        print("Distance quantiles:", np.quantile(distances, [0.25, 0.5, 0.75]))

        plt.hist(distances, bins=30, alpha=0.5, label=label)

    plt.title('Histogram of Distances by Simplex Configuration')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend(title="Simplex Configurations", title_fontsize='13', fontsize='11')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Example of usage
    embeddings = np.random.rand(100, 5)  # Generate some random embeddings
    ratios = [0.3, 0.5, 0.7]
    k_values = [3, 4, 5]
    eps_grid = np.linspace(0.0001, 0.5, 5)

    splits = split_data(embeddings, ratios)
    simplexes = build_simplexes(splits[0.3][0], k_values, eps_grid)  # Using the 0.3 train split as an example
    results = project_points(splits[0.3][1], simplexes)
    analyze_results(results)
