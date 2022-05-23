import numpy as np
from simpy import Environment

# import for intern files
from simulation import config
from simulation.util import print_verbose, print_silent
from simulation.requests import UsageRequest, FloorRequest


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
                 skyscraper,
                 starting_floor: int,
                 destination_floor: int):
        self.passenger_id = passenger_id
        self.skyscraper = skyscraper
        self.__environment = environment
        # if neither start nor destination is provided
        if starting_floor is None and destination_floor is None:
            self.starting_floor, self.destination_floor = generate_random_start_and_destination()
        # if starting floor is provided but no destination
        elif starting_floor is not None and destination_floor is None:
            self.starting_floor, self.destination_floor = [starting_floor,
                                                           generate_random_floor(exclude=starting_floor)]
        # if destination floor is provided but no start
        elif starting_floor is None and destination_floor is not None:
            self.starting_floor, self.destination_floor = [generate_random_floor(exclude=destination_floor),
                                                           destination_floor]
        else:
            self.starting_floor = starting_floor
            self.destination_floor = destination_floor
        self.skyscraper.passenger_route_log.append([self.starting_floor, self.destination_floor])
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
        start_queue_time = self.__environment.now
        if self.starting_floor < self.destination_floor:
            yield self.skyscraper.floor_list[self.starting_floor].queue_up.put(usage_request)
        else:
            yield self.skyscraper.floor_list[self.starting_floor].queue_down.put(usage_request)

        elevator_id = yield request_flag
        end_queue_time = self.__environment.now
        self.debug_log(f'{elevator_id} accepted transportation request')

        # Request the elevator to hold at your destination
        start_time_transportation = self.__environment.now
        floor_flag = self.__environment.event()
        floor_request = FloorRequest(floor_flag, self.destination_floor)
        yield self.skyscraper.elevator_list[elevator_id].passenger_requests.put(floor_request)
        yield floor_flag

        end_time = self.__environment.now
        total_time_waited = (end_time - start_time_total) * config.SECONDS_PER_STEP
        transportation_time = (end_time - start_time_transportation) * config.SECONDS_PER_STEP
        self.debug_log(f'Zieletage erreicht, insgesamt gewartet: {total_time_waited:.2f} Sekunden \n'
                       f'davon im Aufzug: {transportation_time:.2f}')
        self.skyscraper.total_time_log.append(total_time_waited)
        self.skyscraper.queue_time_log.append((end_queue_time- start_queue_time) * config.SECONDS_PER_STEP)
        self.skyscraper.travel_time_log.append(transportation_time)



