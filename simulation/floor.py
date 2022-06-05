from simpy import Environment, Store

# import for intern files
from simulation import config


class Floor:
    def __init__(self, environment: Environment, floor_number: int):
        # floor which is represented by this object
        self.floor_number = floor_number
        # Queues representing waiting passengers on this floor
        # only create a queue for "up" if it is the ground floor
        if floor_number == 0:
            self.queue_up = Store(environment)
            self.queue_down = None
        # only create a queue for "down" if it is the top floor
        elif floor_number == config.NUM_OF_FLOORS-1:
            self.queue_up = None
            self.queue_down = Store(environment)
        else:
            self.queue_up = Store(environment)
            self.queue_down = Store(environment)

    @property
    def num_waiting_up(self):
        """
        Returns the amount of people currently waiting on this floor for the elevator to
        reach a higher floor

        :return int:
        """
        if self.queue_up is None:
            return 0
        return len(self.queue_up.items)

    @property
    def num_waiting_down(self):
        """
        Returns the amount of people currently waiting on this floor for the elevator to
        reach a lower floor

        :return int:
        """
        if self.queue_down is None:
            return 0
        return len(self.queue_down.items)
