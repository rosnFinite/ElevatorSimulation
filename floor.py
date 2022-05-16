import simpy
import config

class Floor:
    def __init__(self, environment, floor_number: int):
        # floor which is represented by this object
        self.floor_number = floor_number
        # Queues representing waiting passengers on this floor
        # only create a queue for "up" if it is the ground floor
        if floor_number == 0:
            self.queue_up = simpy.Store(environment)
            self.queue_down = None
        # only create a queue for "down" if it is the top floor
        elif floor_number == config.NUM_OF_FLOORS-1:
            self.queue_up = None
            self.queue_down = simpy.Store(environment)
        else:
            self.queue_up = simpy.Store(environment)
            self.queue_down = simpy.Store(environment)

    def num_waiting_up(self) -> int:
        """
        Returns the amount of people currently waiting for the elevator to
        reach a higher floor

        :return int: amount of people waiting in "up" queue
        """
        if self.queue_up is None:
            return -1
        return len(self.queue_up.items)

    def num_waiting_down(self) -> int:
        """
        Returns the amount of people currently waiting for the elevator to
        reach a lower floor

        :return int:
        """
        if self.queue_down is None:
            return -1
        return len(self.queue_down.items)

    def request_up(self, usage_request):
        """
        Requests elevator to go "up".
        Enters the current "up" queue

        :param UsageRequest usage_request:
        :return:
        """
        if self.queue_up is not None:
            self.queue_up.put(usage_request)

    def request_down(self, usage_request):
        """
        Requests elevator to go "down".
        Enters the current "down" queue

        :param UsageRequest usage_request:
        :return:
        """
        if self.queue_down is not None:
            self.queue_down.put(usage_request)
