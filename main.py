# Loading project modules
from vispy import app
from Vesicle import Vesicles
from Cell import Cell
from Visualizer import Visualizer

# Simulation Parameters
GRID_SIZE_X = 10
GRID_SIZE_Y = 10
L = 10.0
CIRCLE_RADIUS = 10.0
DIFFUSION_COEFF = 5.0
DT = 0.1
N_SAMPLES = 8
N_VESICLES = 4

# Initialize simulation components
cell = Cell(GRID_SIZE_Y, GRID_SIZE_X, L)
animation = Visualizer(cell.x_max, cell.y_max, L)
vesicles = Vesicles()

# Seed initial vesicles
for _ in range(N_VESICLES):
    vesicles.create(CIRCLE_RADIUS, cell.x_max, cell.y_max, N_SAMPLES, DIFFUSION_COEFF, DT, cell)

# Draw initial scene
animation.background_init(cell.triangles)
animation.draw_vesicles(vesicles.vesicles)

# Update function
def update(event):
    animation.reset_background(cell)
    vesicles.diffuse(cell)
    animation.refresh(vesicles.vesicles, cell)

# Run animation
timer = app.Timer(interval=0.05, connect=update, start=True)

if __name__ == '__main__':
    app.run()
