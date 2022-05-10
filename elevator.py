import config
from floor import Floor


class Elevator:
    def __init__(self, environment, starting_floor: int, floor_list: list[Floor]):
        self.__environment = environment
        self.current_floor = starting_floor
        self.__num_of_passengers = config.ELEVATOR_PAYLOAD
        self.__floor_list = floor_list
        self.floor_queue_up = floor_list[starting_floor].queue_up
        self.floor_queue_down = floor_list[starting_floor].queue_down
