#!/Users/mip182/opt/anaconda3/bin/python3

import numpy as np
import cvxpy as cp



class SimplexProjection:
    def __init__(self, simplexes, eps=1e-9):
        """
        Initializes the class with a list of simplexes.
        :param simplexes: List of np.array, where each array represents a simplex vertices.
        """
        self.simplexes = simplexes
        self.eps = eps

    def project_point_to_simplex(self, y, vertices):
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
        return x.value

    def find_closest_projection(self, points):
        """
        For each point in points, finds the closest projection among the projections on the simplixes.
        :param points: np.array - Array of d-dimensional vectors.
        :return: List of np.array - Closest projections for each point.
        :return: List of int - The projection simplex index for each point.
        """
        closest_projections = []
        projection_simplex = []
        for point in points:
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
            closest_projections.append(closest)
            projection_simplex.append(simplex_ind)
        return closest_projections, projection_simplex

    def is_projections_in_simplexes(self, points):
        """
        Checks whether the projections of the points on the closest simplexes are the same as the original points within a specified epsilon.
        :param points: np.array - Array of d-dimensional vectors.
        :param eps: float - Tolerance for checking equality.
        :return: np.array - Array of booleans.
        """
        projections = self.find_closest_projection(points)
        return np.array([np.linalg.norm(point - proj) < self.eps for point, proj in zip(points, projections)])


# Example usage
simplices = np.random.rand(5, 4, 3)  # List of simplices with random vertices
points = np.random.rand(20, 3)  # Set of points
projector = SimplexProjection(simplices)
closest_projections = projector.find_closest_projection(points)
zero_one_array, projection_indexes = projector.is_projections_in_simplexes(points)
print("Points: ", points)
print("Closest Projections:", closest_projections)
print("Projection Check (True or False):", zero_one_array)
print("Projection Indexes:", projection_indexes)
