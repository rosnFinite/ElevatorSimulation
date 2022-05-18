"""
This files provides globally usable configurations
for the simulations.
"""

# Top Level configurations
SKYSCRAPER_CAPACITY = 500
SECONDS_PER_STEP = 10
SIMULATION_TIME = int(24*60*(60/SECONDS_PER_STEP))
NUM_OF_FLOORS = 15
NUM_OF_ELEVATORS = 3
ELEVATOR_PAYLOAD = 5
# Checkpoints to change exp rate and default start and destination floor
EXP_RATE_CHECKPOINTS = {0: [100, None, None],       # scale=100, start=random, destination=random (from 00:00)
                        2520: [2, 0, None],         # scale=4, start=0, destination=random        (from 7:00)
                        2880: [3, None, None],     # scale=12, start=random, destination=random  (from 8:00)
                        4320: [3, None, 0],         # scale=4, start=random, destination=0        (from 12:00)
                        4500: [8, None, None],      # scale=12, start=random, destination=random  (from 12:30)
                        4680: [3, 0, None],         # scale=4, start=0, destination=random        (from 13:00)
                        4860: [3, None, None],        # scale=12, start=0, destination=random       (from 13:30)
                        5760: [2, None, 0],         # scale=4, start=random, destination=0        (from 16:00)
                        6120: [100, None, None]}    # scale=12, start=random, destination=random  (from 17:00)

VERBOSE = False
