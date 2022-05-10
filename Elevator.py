class Elevator:
    def __init__(self, environment, starting_floor: int, floor_list: list, num_of_passengers: int = 5):
        self.__environment = environment
        self.current_floor = starting_floor
        self.__num_of_passengers = num_of_passengers
        self.__floor_list = floor_list
        self.floor_queue_up = floor_list[starting_floor].queue_up
        self.floor_queue_down = floor_list[starting_floor].queue_down
