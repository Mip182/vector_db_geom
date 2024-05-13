#!/Users/mip182/opt/anaconda3/bin/python3

import numpy as np
import cvxpy as cp
from joblib import Parallel, delayed


class SimplexProjection:
    def __init__(self, simplexes, eps=1e-9, is_fast=True):
        """
        Initializes the class with a list of simplexes.
        :param simplexes: List of np.array, where each array represents a simplex vertices.
        """
        self.simplexes = np.array(simplexes)
        self.eps = eps
        self.is_fast = is_fast

    def project_points_to_simplex_fast(self, points, simplex_vertices):
        """
        Projects a batch of points onto a given simplex.

        :param points: np.array of shape (n_points, n_dimensions), points to project.
        :param simplex_vertices: np.array of shape (n_vertices, n_dimensions), vertices of the simplex.
        :return: np.array of shape (n_points, n_dimensions), projected points.
        """
        n_points = points.shape[0]
        n_vertices = simplex_vertices.shape[0]

        lambdas = cp.Variable((n_points, n_vertices))
        simplex_matrix = cp.Parameter((n_vertices, points.shape[1]))
        simplex_matrix.value = simplex_vertices

        constraints = [
            cp.sum(lambdas, axis=1) == 1,
            lambdas >= 0
        ]

        projected_points = lambdas @ simplex_matrix
        objective = cp.Minimize(cp.sum_squares(points - projected_points))

        problem = cp.Problem(objective, constraints)
        problem.solve()

        return projected_points.value

    def find_closest_projections_fast(self, points, simplex_set=None):
        """
        Finds the closest projections of a set of points onto multiple simplexes using parallel computation.

        :param points: np.array of shape (n_points, n_dimensions)
            Points to project onto the simplexes.
        :param simplex_set: list of lists or np.array, optional
            Set of simplexes for projection. Defaults to self.simplexes if None.

        :return: np.array of shape (n_points, n_dimensions)
            Closest projections of the input points.
        """
        if simplex_set is None:
            simplex_set = self.simplexes

        n_points = points.shape[0]
        closest_projections = np.zeros_like(points)
        min_distances = np.full(n_points, np.inf)

        def process_simplex(simplex_vertices):
            projected_points = self.project_points_to_simplex_fast(points, np.array(simplex_vertices))
            if projected_points is None:
                return None, None
            distances = np.linalg.norm(points - projected_points, axis=1)
            return distances, projected_points

        results = Parallel(n_jobs=-1)(delayed(process_simplex)(simplex_vertices) for simplex_vertices in simplex_set)

        for distances, projected_points in results:
            if distances is None:
                continue
            closer_indices = distances < min_distances
            min_distances[closer_indices] = distances[closer_indices]
            closest_projections[closer_indices] = projected_points[closer_indices]

        return closest_projections

    def project_point_to_simplex_slow(self, y, vertices):
        """
        Project a point onto a simplex defined by its vertices.
        :param y: np.array - The point to be projected.
        :param vertices: np.array - The vertices of the simplex.
        :return: np.array - The projected point on the simplex.
        """
        k = vertices.shape[0]
        alphas = cp.Variable(k)
        x = vertices.T @ alphas
        objective = cp.Minimize(cp.norm(y - x, 2))
        constraints = [cp.sum(alphas) == 1, alphas >= 0]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        return x.value.reshape(-1)

    def find_closest_projections_slow(self, points, simplex_set=None):
        """
        Finds the closest projections of points onto multiple simplexes.

        :param points: np.array
            Array of d-dimensional vectors to be projected.
        :param simplex_set: list of lists or np.array, optional
            Set of simplexes for projection. Defaults to self.simplexes if None.

        :return: tuple
            - np.array: Closest projections for each point.
            - np.array: Indices of the simplex for each closest projection.
        """
        if simplex_set is None:
            simplex_set = self.simplexes

        def process_point(point):
            closest = None
            min_dist = float('inf')
            simplex_ind = None
            for ind, simplex in enumerate(simplex_set):
                projected = self.project_point_to_simplex(point, simplex)
                dist = np.linalg.norm(point - projected)
                if dist < min_dist:
                    closest = projected
                    min_dist = dist
                    simplex_ind = ind
                    if min_dist < self.eps:
                        break
            return closest, simplex_ind

        results = Parallel(n_jobs=-1)(delayed(process_point)(point) for point in points)

        closest_projections, projection_simplex_indices = zip(*results)

        return np.vstack(closest_projections), np.array(projection_simplex_indices)

    def find_closest_projections(self, points, simplex_set=None):
        """
        Finds the closest projections of points onto multiple simplexes.

        Uses a fast or slow method based on the value of is_fast.

        :param points: np.array
            Array of d-dimensional vectors to be projected.
        :param simplex_set: list of lists or np.array, optional
            Set of simplexes for projection. Defaults to self.simplexes if None.

        :return: np.array
            Closest projections for each point.
        """
        if self.is_fast:
            return self.find_closest_projections_fast(points, simplex_set)
        else:
            closest_projections, _ = self.find_closest_projections_slow(points, simplex_set)
            return closest_projections

    def is_projections_in_simplexes(self, points):
        """
        Checks whether the projections of the points on the closest simplexes are the same as the original points within a specified epsilon.
        :param points: np.array - Array of d-dimensional vectors.
        :param eps: float - Tolerance for checking equality.
        :return: np.array - Array of booleans.
        """
        projections = self.find_closest_projections(points)
        return np.array([np.linalg.norm(point - proj) < self.eps for point, proj in zip(points, projections)])

    def sample_simplexes(self, share):
        """
        Selects a random sample of the given share of simplexes from self.simplexes.

        :param share: float
            The fraction of simplexes to sample (e.g., 0.1 for 10%).

        :return: list
            A random sample of simplexes.
        """
        if not 0 < share <= 1:
            raise ValueError("Share must be a float between 0 and 1.")

        n_simplexes = len(self.simplexes)
        sample_size = int(n_simplexes * share)
        sampled_indices = np.random.choice(n_simplexes, sample_size, replace=False)

        return [self.simplexes[i] for i in sampled_indices]


if __name__ == "__main__":
    # Example usage
    simplices = np.random.rand(5, 4, 3)  # List of simplices with random vertices
    points = np.random.rand(20, 3)  # Set of points
    projector = SimplexProjection(simplices)
    closest_projections = projector.find_closest_projections(points)
    zero_one_array = projector.is_projections_in_simplexes(points)
    print("Points: ", points)
    print("Closest Projections:", closest_projections)
    print("Projection Check (True or False):", zero_one_array)
