class UsageRequest:
    def __init__(self, event, request_time):
        self.__completion_event = event
        self.request_time = request_time

    def accept_usage_request(self, elevator_id):
        self.__completion_event.succeed(value=elevator_id)


class FloorRequest:
    def __init__(self, event, destination_floor, request_time):
        self.__completion_event = event
        self.__destination_floor = destination_floor
        self.request_time = request_time

    @property
    def destination_floor(self):
        if self.__destination_floor is not None:
            return self.__destination_floor

    def reached_floor(self):
        self.__completion_event.succeed()

