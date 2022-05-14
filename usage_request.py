class UsageRequest:
    def __init__(self, event, user):
        self.__completion_event = event
        self.__userId = user

    def fullfill_usage_request(self):
        self.__completion_event.succeed()
