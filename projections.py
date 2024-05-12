#!/Users/mip182/opt/anaconda3/bin/python3

import numpy as np
import cvxpy as cp


class SimplexProjection:
    def __init__(self, simplexes, eps=1e-9):
        """
        Initializes the class with a list of simplexes.
        :param simplexes: List of np.array, where each array represents a simplex vertices.
        """
        self.simplexes = np.array(simplexes)
        self.eps = eps

    def project_points_to_simplex(self, points, simplex_vertices):
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

    def find_closest_projections(self, points):
        """
        Finds the closest projections of a set of points onto multiple simplexes.

        :param points: np.array of shape (n_points, n_dimensions), points to project.
        :return: np.array of shape (n_points, n_dimensions), closest projections.
        """
        n_points = points.shape[0]
        closest_projections = np.zeros_like(points)
        min_distances = np.full(n_points, np.inf)

        for simplex_vertices in self.simplexes:
            projected_points = self.project_points_to_simplex(points, np.array(simplex_vertices))
            distances = np.linalg.norm(points - projected_points, axis=1)

            # Update closest projections
            closer_indices = distances < min_distances
            min_distances[closer_indices] = distances[closer_indices]
            closest_projections[closer_indices] = projected_points[closer_indices]

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
