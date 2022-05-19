class UsageRequest:
    def __init__(self, event):
        self.__completion_event = event

    def accept_usage_request(self, elevator_id):
        self.__completion_event.succeed(value=elevator_id)


class FloorRequest:
    def __init__(self, event, destination_floor):
        self.__completion_event = event
        self.__destination_floor = destination_floor

    @property
    def destination_floor(self):
        if self.__destination_floor is not None:
            return self.__destination_floor

    def reached_floor(self):
        self.__completion_event.succeed()

