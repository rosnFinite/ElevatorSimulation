import simpy


class Floor:
    def __init__(self, environment, floor_number: int):
        # floor which is represented by this object
        self.floor_number = floor_number
        # Queues representing waiting passengers on this floor
        # only create a queue for "up" if it is the ground floor
        if floor_number == 0:
            self.__queue_up = simpy.Store(environment)
            self.__queue_down = None
        # only create a queue for "down" if it is the top floor
        elif floor_number == 14:
            self.__queue_up = None
            self.__queue_down = simpy.Store(environment)
        else:
            self.__queue_up = simpy.Store(environment)
            self.__queue_down = simpy.Store(environment)

    @property
    def queue_up(self):
        return self.__queue_up

    @property
    def queue_down(self):
        return self.__queue_down
