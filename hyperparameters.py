
NET = "indigooryx"
LEARNING_RATE = 1e-2
NUM_EPISODES = 275
TIMESTEP_LENGTH = 1 / 120.0  # Duration (in hours) of each timestep
BATCH_SIZE = 128
TARGET_UPDATE = 100
GAMMA = 0.999

# Distance between values of grid (in coordinates)
BD_COORD = 0.5
GRID_DIM = 25

# Airplane info
FUEL_HOURS = 3
MAX_CRUISING_SPEED = 575
ALTITUDE = 21000

# Epsilon greedy values (start, end, decay rate)
EPS_START = 0.90
EPS_END = 0.05
EPS_DECAY = 1000  # 200

DATETIME = ['2018072203']

assert (GRID_DIM % 2 == 1) and (GRID_DIM > 1), "GRID_DIM must be an odd " \
    + "integer greater than one!"
