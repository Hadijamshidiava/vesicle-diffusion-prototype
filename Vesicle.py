from dataclasses import dataclass
import numpy as np

@dataclass
class Vesicle:
    """
    Represents a single vesicle with a center, samples, and diffusion properties.
    
    Attributes:
        center (np.ndarray): 2D coordinate of the vesicle center.
        samples (np.ndarray): Coordinates of sampled points on the vesicle perimeter.
        diffusion_coeff (float): Diffusion coefficient (D).
        nSample (int): Number of perimeter samples.
        dt (float): Time step used for diffusion.
        r (float): Vesicle radius.
    """
    center: np.ndarray
    samples: np.ndarray
    diffusion_coeff: float
    nSample: int
    dt: float
    r: float


class Vesicles:
    """
    A container class for managing and diffusing multiple vesicles in a 2D cell mesh.
    """

    def __init__(self):
        self.vesicles = []

    def is_position_valid(self, center: np.ndarray, samples: np.ndarray, cell) -> bool:
        """
        Checks whether a vesicle at the given center and samples can be placed without
        overlapping occupied triangles.

        Args:
            center (np.ndarray): Center of the vesicle.
            samples (np.ndarray): Sample points on its perimeter.
            cell (Cell): The cell mesh.

        Returns:
            bool: True if the position is valid; False otherwise.
        """
        # Compute distance from vesicle center to each triangle center
        diff = cell.centers - center
        distances = np.linalg.norm(diff, axis=1)
        triangleIndexes = np.where(distances <= 10)[0]

        if triangleIndexes.size == 0:
            return False

        for index in triangleIndexes:
            isOc = cell.is_occupied(samples, index)
            if cell.triangles[index].occupied and isOc:
                return False
            elif isOc:
                cell.triangles[index].occupied = True

        return True

    def create(self, CIRCLE_RADIUS, x_max, y_max, N_SAMPLES, diffusion_coeff, dt, cell):
        """
        Attempts to create a new vesicle at a random valid position.

        If the generated position is invalid (due to overlap), it retries recursively.

        Args:
            CIRCLE_RADIUS (float): Radius of each vesicle.
            x_max, y_max (float): Simulation area boundaries.
            N_SAMPLES (int): Number of sample points on vesicle perimeter.
            diffusion_coeff (float): Diffusion coefficient.
            dt (float): Time step.
            cell (Cell): The simulation mesh.
        """
        center = np.random.uniform(
            [2*CIRCLE_RADIUS, CIRCLE_RADIUS],
            [x_max - 2*CIRCLE_RADIUS, y_max - CIRCLE_RADIUS]
        )

        angles = np.linspace(0, 2*np.pi, N_SAMPLES, endpoint=False)
        samples = center + CIRCLE_RADIUS * np.column_stack((np.cos(angles), np.sin(angles)))

        vesicle = Vesicle(center=center, samples=samples, r=CIRCLE_RADIUS,
                          diffusion_coeff=diffusion_coeff, nSample=N_SAMPLES, dt=dt)

        if self.is_position_valid(center, samples, cell):
            self.vesicles.append(vesicle)
        else:
            # Recursive retry
            self.create(CIRCLE_RADIUS, x_max, y_max, N_SAMPLES, diffusion_coeff, dt, cell)

    def diffuse(self, visual, cell):
        """
        Performs a diffusion step on all vesicles. Each vesicle moves by a random
        Brownian step, and updates its position only if the new location is valid.

        Args:
            visual: Visualization object that holds circles/samples for update.
            cell (Cell): The mesh to test occupancy against.
        """
        for index, vesicle in enumerate(self.vesicles):
            c = vesicle.center.copy()
            s = vesicle.samples.copy()

            # Brownian motion scaling
            scale = np.sqrt(2 * vesicle.diffusion_coeff * vesicle.dt)

            # Random direction (2D)
            displacement = scale * np.random.normal(0, 1, size=2)

            # Apply displacement
            new_center = c + displacement
            new_samples = s + displacement

            # Validate and update
            if self.is_position_valid(new_center, new_samples, cell):
                vesicle.center = new_center
                vesicle.samples = new_samples

