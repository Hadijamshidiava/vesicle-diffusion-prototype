import numpy as np
from vispy import app, scene
from vispy.scene.visuals import Ellipse, Markers, Mesh, Line

class Visualizer:
    """
    Handles 2D visualization of a vesicle diffusion simulation
    using vispy. Responsible for drawing triangles, vesicles,
    and updating visuals in real-time.
    """

    def __init__(self, x_max, y_max, L):
        """
        Initializes the canvas and view.

        Args:
            x_max (float): Maximum X extent of the cell.
            y_max (float): Maximum Y extent of the cell.
            L (float): Grid size to define camera padding.
        """
        canvas = scene.SceneCanvas(keys='interactive', bgcolor='white', size=(900, 900), show=True)
        self.view = canvas.central_widget.add_view()

        self.view.camera = scene.PanZoomCamera(aspect=1)
        self.view.camera.set_range(x=(0, x_max + L), y=(0, y_max + L))

        self.triangle_visuals = []  # List of triangle Mesh visuals
        self.circles = []           # List of vesicle Ellipse visuals

        # Marker visual for sample points on vesicles
        self.sample_markers = Markers(parent=self.view.scene)
        self.sample_markers.set_data(np.zeros((0, 2)), face_color='red', size=5)

    def background_init(self, triangles):
        """
        Initializes the triangle mesh background from Triangle objects.

        Args:
            triangles (list[Triangle]): List of Triangle dataclass instances.
        """
        self.triangle_visuals.clear()
        faces = np.array([[0, 1, 2]])  # Triangle face indices

        for tri in triangles:
            mesh = Mesh(vertices=tri.vertices, faces=faces, color=(1, 0, 0, 0.15), parent=self.view.scene)
            self.triangle_visuals.append(mesh)

    def draw_vesicles(self, vesicles):
        """
        Draws all vesicles as circles and their surface samples as red dots.

        Args:
            vesicles (list[Vesicle]): List of vesicle dataclass instances.
        """
        self.circles.clear()
        allSamples = []

        for vesicle in vesicles:
            circle = Ellipse(
                center=vesicle.center,
                radius=vesicle.r,
                color=(0, 0.5, 1, 0.8),
                border_color='black',
                parent=self.view.scene
            )
            self.circles.append(circle)
            allSamples.append(vesicle.samples)

        allSamples = np.vstack(allSamples)
        self.sample_markers.set_data(allSamples, face_color='red', size=5)

    def refresh(self, vesicles, cell):
        """
        Updates vesicle positions and highlights any triangles marked as occupied.

        Args:
            vesicles (list[Vesicle]): Updated vesicle data.
            cell (Cell): Cell mesh containing occupancy status.
        """
        allSamples = []

        for index, vesicle in enumerate(vesicles):
            self.circles[index].center = vesicle.center
            allSamples.append(vesicle.samples)

        allSamples = np.vstack(allSamples)
        self.sample_markers.set_data(allSamples, face_color='red', size=5)

        for index, triangle in enumerate(cell.triangles):
            if triangle.occupied:
                self.triangle_visuals[index].color = (1, 0, 0, 0.9)

    def reset_background(self, cell):
        """
        Resets triangle colors and clears occupancy state in the mesh.

        Args:
            cell (Cell): The mesh whose triangle states are being reset.
        """
        for index, triangle in enumerate(self.triangle_visuals):
            triangle.color = (1, 0, 0, 0.15)
            # cell.triangles[index].occupied = False
