import numpy as np
import config
from simpy import Environment
from typing import List
from elevator import Elevator
from util import print_verbose, print_silent
from floor import Floor
from requests import UsageRequest, FloorRequest


def generate_random_start_and_destination() -> np.ndarray:
    """
    Generates starting(current)floor and destination floor.
    Guarantees that
    - start != destination
    - 0 <= start/destination < NUM_OF_FLOORS
    """
    return np.random.choice(config.NUM_OF_FLOORS, 2, replace=False)


def generate_random_floor(exclude: int) -> int:
    """
    Generates a random floor for the passenger, either destination or start
    """
    return np.random.choice(np.setdiff1d(range(1, 4), exclude))


class Passenger:
    def __init__(self, passenger_id: int,
                 environment: Environment,
                 floor_list: List[Floor],
                 elevator_list: List[Elevator],
                 time_waited_log: List[float],
                 transportation_time_log: List[float],
                 route_log: List[List[int]],
                 starting_floor: int,
                 destination_floor: int):
        self.__passenger_id = passenger_id
        self.__environment = environment
        self.__elevator_list = elevator_list
        self.__floor_list = floor_list
        self.__total_time_waited_log = time_waited_log
        self.__transportation_time_log = transportation_time_log
        self.__route_log = route_log
        # if neither start nor destination is provided
        if starting_floor is None and destination_floor is None:
            self.__starting_floor, self.__destination_floor = generate_random_start_and_destination()
        # if starting floor is provided but no destination
        elif starting_floor is not None and destination_floor is None:
            self.__starting_floor, self.__destination_floor = [starting_floor,
                                                               generate_random_floor(exclude=starting_floor)]
        # if destination floor is provided but no start
        elif starting_floor is None and destination_floor is not None:
            self.__starting_floor, self.__destination_floor = [generate_random_floor(exclude=destination_floor),
                                                               destination_floor]
        else:
            self.__starting_floor = starting_floor
            self.__destination_floor = destination_floor
        self.__route_log.append([self.__starting_floor, self.__destination_floor])
        self.debug_log = print_silent
        if config.VERBOSE:
            self.debug_log = print_verbose
        self.__environment.process(self.use_elevator())

    def use_elevator(self):
        """
        Process of a passenger using an elevator.
        1. Request an elevator (enter queue for up or down on the current floor, depending on passenger route)
        2. Enter the elevator
        3. Put a hold request for the specified destination inside the elevators passenger_request queue
        """
        start_time_total = self.__environment.now
        request_flag = self.__environment.event()
        usage_request = UsageRequest(request_flag)

        # Enter the correct queue corresponding to start and destination
        current_floor = self.__starting_floor
        destination_floor = self.__destination_floor
        if current_floor < destination_floor:
            yield self.__floor_list[current_floor].queue_up.put(usage_request)
        else:
            yield self.__floor_list[current_floor].queue_down.put(usage_request)

        elevator_id = yield request_flag
        self.debug_log(f'{elevator_id} accepted transportation request')

        # Request the elevator to hold at your destination
        start_time_transportation = self.__environment.now
        floor_flag = self.__environment.event()
        floor_request = FloorRequest(floor_flag, destination_floor)
        yield self.__elevator_list[elevator_id].passenger_requests.put(floor_request)
        yield floor_flag

        end_time = self.__environment.now
        total_time_waited = (end_time - start_time_total) * config.SECONDS_PER_STEP
        transportation_time = (end_time - start_time_transportation) * config.SECONDS_PER_STEP
        self.debug_log(f'Zieletage erreicht, insgesamt gewartet: {total_time_waited:.2f} Sekunden \n'
                       f'davon im Aufzug: {transportation_time:.2f}')
        self.__total_time_waited_log.append(total_time_waited)
        self.__transportation_time_log.append(transportation_time)



