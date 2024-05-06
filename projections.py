#!/Users/mip182/opt/anaconda3/bin/python3

import numpy as np
import cvxpy as cp


class SimplexProjection:
    def __init__(self, simplexes):
        """
        Initializes the class with a list of simplexes.
        :param simplexes: List of np.array, where each array represents a simplex vertices.
        """
        self.simplexes = simplexes

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
        """
        closest_projections = []
        for point in points:
            closest = None
            min_dist = float('inf')
            for simplex in self.simplexes:
                projected = self.project_point_to_simplex(point, simplex)
                dist = np.linalg.norm(point - projected)
                if dist < min_dist:
                    closest = projected
                    min_dist = dist
            closest_projections.append(closest)
        return closest_projections


# Example usage
simplices = [np.random.rand(4, 3), np.random.rand(3, 3)]  # List of simplices with random vertices
points = np.random.rand(5, 3)  # Set of points
projector = SimplexProjection(simplices)
closest_projections = projector.find_closest_projection(points)
print("Points: ", points)
print("Closest Projections:", closest_projections)
