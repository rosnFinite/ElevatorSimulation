import numpy as np
import config
from floor import Floor
from usage_request import UsageRequest


class Passenger:
    def __init__(self, environment, floor_list, passenger_id):
        """

        :param environment:
        :type: simpy.core.Environment
        :param floor_list:
        :type: list[Floor]
        :param passenger_id:
        :type: int
        """
        self.__environment = environment
        self.__floor_list = floor_list
        self.__passenger_id = passenger_id
        self.__starting_floor, self.__destination_floor = self.__generate_start_and_destination()
        self.__time_waited = 0
        self.__environment.process(self.use_elevator())

    @property
    def starting_floor(self) -> int:
        """
        Get starting floor of the passenger

        :return: starting floor of the passenger
        :type: int
        """
        return self.__starting_floor

    @property
    def destination_floor(self) -> int:
        return self.__destination_floor

    def __generate_start_and_destination(self) -> np.ndarray:
        """
        Generates starting(current) floor and destination floor
        Gurantees that
            - start != destination
            - 0 <= start/destination < NUM_OF_FLOORS

        :return:
        :type: numpy.ndarray[int]
        """
        return np.random.choice(config.NUM_OF_FLOORS, 2, replace=False)

    def use_elevator(self):
        """
        Request Elevator to reach destination floor from current floor.
        Enters queue_up of current floor if current_floor < destination_floor
        otherwise queue_down

        :return:
        """
        request_flag = self.__environment.event()
        usage_request = UsageRequest(request_flag, self.__passenger_id)

        # Decide for the correct queue to put the request in (UP or DOWN)
        current_floor = self.__starting_floor
        destination_floor = self.__destination_floor
        if current_floor < destination_floor:
            yield self.__floor_list[current_floor].queue_up.put(usage_request)
        else:
            yield self.__floor_list[current_floor].queue_down.put(usage_request)

