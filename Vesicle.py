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
    overlapped: list


class Vesicles:
    """
    A container class for managing and diffusing multiple vesicles in a 2D cell mesh.
    """

    def __init__(self):
        self.vesicles = []

    def overlap_check(self, r, oldOverlapped: list, center: np.ndarray, samples: np.ndarray, cell) -> bool:
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
        newOverlapped = []
        diff = cell.centers - center
        distances = np.linalg.norm(diff, axis=1)
        triangleIndexes = np.where(distances <= 3*r)[0]

        if triangleIndexes.size == 0:
            return (False,oldOverlapped)

        for index in triangleIndexes:
            isOc = cell.is_occupied(center, r,samples, index)
            if cell.triangles[index].occupied and isOc and not index in oldOverlapped :
                return (False,oldOverlapped)
            elif isOc:
                newOverlapped.append(index)
            
        for index in oldOverlapped:
            cell.triangles[index].occupied = False
        for index in newOverlapped:
            cell.triangles[index].occupied = True

        return (True,newOverlapped)

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
                          diffusion_coeff=diffusion_coeff, nSample=N_SAMPLES, dt=dt, overlapped=[])

        flag,overlap = self.overlap_check(CIRCLE_RADIUS, [], center, samples, cell)
        if flag:
            vesicle.overlapped = overlap
            self.vesicles.append(vesicle)
        else:
            # Recursive retry
            self.create(CIRCLE_RADIUS, x_max, y_max, N_SAMPLES, diffusion_coeff, dt, cell)

    def diffuse(self, cell):
        """
        Performs a diffusion step on all vesicles. Each vesicle moves by a random
        Brownian step, and updates its position only if the new location is valid.

        Args:
            cell (Cell): The mesh to test occupancy against.
        """
        for index, vesicle in enumerate(self.vesicles):
            old_center = vesicle.center.copy()
            old_samples = vesicle.samples.copy()
            old_overlap = vesicle.overlapped.copy()

            # Brownian motion scaling
            scale = np.sqrt(2 * vesicle.diffusion_coeff * vesicle.dt)

            # Random direction (2D)
            displacement = scale * np.random.normal(0, 1, size=2)

            # Apply displacement
            new_center = old_center + displacement
            new_samples = old_samples + displacement

            # Validate and update
            flag,new_overlap = self.overlap_check(vesicle.r, old_overlap, new_center, new_samples, cell)
            if flag:
                vesicle.center = new_center
                vesicle.samples = new_samples
                vesicle.overlapped = new_overlap

