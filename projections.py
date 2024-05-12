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

    def find_closest_projections_fast(self, points):
        """
        Finds the closest projections of a set of points onto multiple simplexes.

        :param points: np.array of shape (n_points, n_dimensions), points to project.
        :return: np.array of shape (n_points, n_dimensions), closest projections.
        """
        n_points = points.shape[0]
        closest_projections = np.zeros_like(points)
        min_distances = np.full(n_points, np.inf)

        def process_simplex(simplex_vertices):
            projected_points = self.project_points_to_simplex_fast(points, np.array(simplex_vertices))
            if projected_points is None:
                return None, None
            distances = np.linalg.norm(points - projected_points, axis=1)
            return distances, projected_points

        results = Parallel(n_jobs=-1)(delayed(process_simplex)(simplex_vertices) for simplex_vertices in self.simplexes)

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

    def find_closest_projections_slow(self, points):
        """
        For each point in points, finds the closest projection among the projections on the simplices.
        :param points: np.array - Array of d-dimensional vectors.
        :return: List of np.array - Closest projections for each point.
        :return: List of int - The projection simplex index for each point.
        """

        def process_point(point):
            closest = None
            min_dist = float('inf')
            simplex_ind = None
            for ind, simplex in enumerate(self.simplexes):
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

    def find_closest_projections(self, points):
        """
        For each point in points, finds the closest projection among the projections on the simplixes.
        If is_fast is True, then run fast version, otherwise run slow version.
        :param points: np.array - Array of d-dimensional vectors.
        :return: List of np.array - Closest projections for each point.
        """
        if self.is_fast:
            return self.find_closest_projections_fast(points)
        else:
            closest_projections, _ = self.find_closest_projections_slow(points)
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
