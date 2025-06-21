from dataclasses import dataclass
import numpy as np
from vispy.scene.visuals import Ellipse, Markers, Mesh, Line


@dataclass
class Triangle:
    """
    Represents a single triangle in the 2D mesh of the cell membrane.

    Attributes:
        vertices (np.ndarray): 3x2 array representing the coordinates of triangle vertices.
        normals (np.ndarray): 3x2 array of normalized edge outward-facing vectors (for side-test).
        occupied (bool): Indicates whether the triangle is currently occupied by a vesicle.
    """
    vertices: list[int]
    normals: list[int]
    occupied: bool

class Cell():
    """
    Models the 2D cell body as a hexagonally packed mesh of triangles.
    
    This is a simplification of the 3D model (which would use tetrahedrons),
    suitable for 2D visualization and prototyping diffusion/spatial logic.
    """
    def __init__(self, GRID_SIZE_Y: int, GRID_SIZE_X: int, L: float):
        """
        Initializes the cell mesh.

        Args:
            GRID_SIZE_Y (int): Number of rows in the grid.
            GRID_SIZE_X (int): Number of columns in the grid.
            L (float): Side length of each triangle unit.
        """
        self.points = self._generate_grid(GRID_SIZE_Y, GRID_SIZE_X, L)
        self.triangles = []
        self.centers = []

        self._generate_mesh(GRID_SIZE_Y, GRID_SIZE_X)

    def _generate_grid(self, rows, cols, L):
        """
        Generates a hexagonally packed 2D grid of points.
        
        Returns:
            np.ndarray: Array of shape (N_points, 2) containing vertex coordinates.
        """
        points = []
        for j in range(rows):
            for i in range(cols):
                x = i * L + (j % 2) * (L / 2)
                y = j * (np.sqrt(3) / 2) * L
                points.append([x, y])
        return np.array(points)

    def _generate_mesh(self, rows, cols):
        """
        Builds the triangle mesh by dividing quads into two triangles.
        """
        def create_triangle(vertices: np.ndarray):
            normals = []
            center = np.zeros(2)
            for i in range(3):
                p_i = vertices[i]
                p_j = vertices[(i + 1) % 3]
                edge = p_j - p_i
                normal = np.array([-edge[1], edge[0]])
                normal /= np.linalg.norm(normal)
                normals.append(normal)
                center += p_i
            triangle = Triangle(vertices=vertices, normals=np.array(normals), occupied=False)
            self.triangles.append(triangle)
            self.centers.append(center / 3)

        for j in range(rows - 1):
            for i in range(cols - 1):
                idx = j * cols + i
                p0 = self.points[idx]
                p1 = self.points[idx + 1]
                p2 = self.points[idx + cols]
                p3 = self.points[idx + cols + 1]

                if (j % 2) == 0:
                    create_triangle(np.array([p0, p1, p2]))
                    create_triangle(np.array([p1, p3, p2]))
                else:
                    create_triangle(np.array([p0, p1, p3]))
                    create_triangle(np.array([p0, p3, p2]))

    def is_occupied(self, center, r, samples: np.ndarray, triIndex: int) -> bool:
        """
        Checks whether any of the sample points fall within the triangle at triIndex.

        Args:
            samples (np.ndarray): Array of shape (N_samples, 2) containing sample points.
            triIndex (int): Index of the triangle to check.

        Returns:
            bool: True if any sample is inside the triangle.
        """
        triangle = self.triangles[triIndex]
        s = samples[None, :, :]                    # shape: (1, N, 2)
        v = triangle.vertices[:, None, :]          # shape: (3, 1, 2)
        diff = s - v                               
        dot = np.sum(diff * triangle.normals[:, None, :], axis=-1)
        inside = np.all(dot >= 0, axis=0)          # shape: (N,)
        diff = v - center
        distances = np.linalg.norm(diff, axis=1)
        trianglesInside = np.all(distances <= r, axis=1)
        return (np.any(inside) or np.any(trianglesInside))

    @property
    def x_max(self) -> float:
        """Returns maximum X extent of the grid."""
        return np.max(self.points[:, 0])

    @property
    def y_max(self) -> float:
        """Returns maximum Y extent of the grid."""
        return np.max(self.points[:, 1])

