"""
Function to translate user defined behaviour to a format used by the simulation instance
"""
import numpy as np


def create_passenger_behaviour(seconds_per_step, spawn_behaviour, floor_behaviour):
    """
    Changes nine checkpoints (predefined amount of checkpoints) to match the behaviour specified by the user

    :param int seconds_per_step:
    :param str spawn_behaviour:
    :param str floor_behaviour:
    :return: Dict[int, tuple(int,int)]
    """
    default_spawnrates = [100, 2, 3, 3, 8, 3, 3, 2, 100]
    default_start_dest = [(None, None),
                          (0, None),
                          (None, None),
                          (None, 0),
                          (None, None),
                          (0, None),
                          (None, None),
                          (None, 0),
                          (None, None)
                          ]
    default_time = [0, 7, 8, 12, 12.5, 13, 13.5, 16, 17]
    spawnrate_checkpoints = {}
    if spawn_behaviour == "realism":
        for index, spawnrate in enumerate(default_spawnrates):
            spawnrate_checkpoints[int(default_time[index] * 60 * (60 / seconds_per_step))] = [spawnrate]
    if spawn_behaviour == "random":
        random_spawnrates = np.random.randint(2, 20, size=9)
        random_time = [0, 3, 6, 9, 12, 15, 18, 21, 23]
        for index, spawnrate in enumerate(random_spawnrates):
            spawnrate_checkpoints[int(random_time[index] * 60 * (60 / seconds_per_step))] = [int(spawnrate)]
    if spawn_behaviour == "static":
        for time in default_time:
            spawnrate_checkpoints[int(time * 60 * (60 / seconds_per_step))] = [3]

    if floor_behaviour == "realism":
        for index, cp in enumerate(spawnrate_checkpoints):
            spawnrate_checkpoints[cp].append(default_start_dest[index][0])
            spawnrate_checkpoints[cp].append(default_start_dest[index][1])
    if floor_behaviour == "random":
        for index, cp in enumerate(spawnrate_checkpoints):
            spawnrate_checkpoints[cp].append(None)
            spawnrate_checkpoints[cp].append(None)
    if floor_behaviour == "static":
        for index, cp in enumerate(spawnrate_checkpoints):
            spawnrate_checkpoints[cp].append(0)
            spawnrate_checkpoints[cp].append(None)

    return spawnrate_checkpoints
