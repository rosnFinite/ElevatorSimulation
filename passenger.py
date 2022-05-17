import numpy as np
import config
from simpy import Environment
from typing import List
from elevator import Elevator
from util import print_verbose, print_silent
from floor import Floor
from requests import UsageRequest, FloorRequest


class Passenger:
    def __init__(self, passenger_id: int,
                 environment: Environment,
                 floor_list: List[Floor],
                 elevator_list: List[Elevator], time_waited_log: List[float]):
        self.__passenger_id = passenger_id
        self.__environment = environment
        self.__elevator_list = elevator_list
        self.__floor_list = floor_list
        self.__time_waited_log = time_waited_log
        self.__starting_floor, self.__destination_floor = [0, self.__generate_destination()]
        self.__time_waited = 0
        self.__environment.process(self.use_elevator())

    @property
    def starting_floor(self) -> int:
        """
        Get starting floor of the passenger

        :return int: starting floor of the passenger
        """
        return self.__starting_floor

    @property
    def destination_floor(self) -> int:
        """
        Get the destination floor of the passenger instance
        """
        return self.__destination_floor

    def __generate_start_and_destination(self) -> np.ndarray:
        """
        Generates starting(current)floor and destination floor.
        Guarantees that
        - start != destination
        - 0 <= start/destination < NUM_OF_FLOORS
        """
        return np.random.choice(config.NUM_OF_FLOORS, 2, replace=False)

    def __generate_destination(self) -> int:
        """
        Generates a destination floor for the passenger
        """
        return np.random.choice(range(1, config.NUM_OF_FLOORS))

    def use_elevator(self):
        """
        Request Elevator to reach destination floor from current floor.
        Enters queue_up of current floor if current_floor < destination_floor
        otherwise queue_down
        """
        if config.VERBOSE:
            log = print_verbose
        else:
            log = print_silent
        start_time = self.__environment.now
        request_flag = self.__environment.event()
        usage_request = UsageRequest(request_flag, self.__passenger_id)

        # Decide for the correct queue to put the request in (UP or DOWN)
        current_floor = self.__starting_floor
        destination_floor = self.__destination_floor
        if current_floor < destination_floor:
            yield self.__floor_list[current_floor].queue_up.put(usage_request)
        else:
            yield self.__floor_list[current_floor].queue_down.put(usage_request)

        elevator_id = yield request_flag
        log(f'{elevator_id} accepted transportation request')

        floor_flag = self.__environment.event()
        floor_request = FloorRequest(floor_flag, destination_floor)
        yield self.__elevator_list[elevator_id].passenger_requests.put(floor_request)
        yield floor_flag
        self.__time_waited = (self.__environment.now - start_time) * 10
        log(f'Zieletage erreicht, insgesamt gewartet: {self.__time_waited:.2f} Sekunden')
        self.__time_waited_log.append(self.__time_waited)



