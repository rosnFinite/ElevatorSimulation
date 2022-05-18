import simpy
import config
from typing import List
from util import print_verbose, print_silent
from floor import Floor


class Elevator:
    def __init__(self, elevator_id: int, environment: simpy.Environment, floor_list: List[Floor], starting_floor: int):
        self.id = elevator_id
        self.__environment = environment
        self.current_floor = starting_floor
        self.__floor_list = floor_list
        self.passenger_requests = simpy.Store(environment, capacity=config.ELEVATOR_PAYLOAD)
        self.__num_of_passengers = 0
        self.__direction = 1

    def accept_passengers(self):
        # print(self.__floor_list[self.current_floor].num_waiting_up())
        # as long as there is enough room, accept passengers
        if config.VERBOSE:
            log = print_verbose
        else:
            log = print_silent
        accepted = False
        log(f'Aufzug ID: {self.id}')
        log(f'Etage: {self.current_floor}')
        log(f'Richtung: {self.__direction}')
        log(f'Wartend hoch: {self.__floor_list[self.current_floor].num_waiting_up()}')
        log(f'Wartend runter: {self.__floor_list[self.current_floor].num_waiting_down()}')

        waiting_up = self.__floor_list[self.current_floor].num_waiting_up()
        waiting_down = self.__floor_list[self.current_floor].num_waiting_down()

        if self.__direction == 1 and waiting_up > 0:
            accepted = True
            while self.__num_of_passengers < config.ELEVATOR_PAYLOAD and waiting_up > 0:
                request = yield self.__floor_list[self.current_floor].queue_up.get()
                request.accept_usage_request(self.id)
                self.__num_of_passengers += 1
                log(
                    f'Fahrgast aufgenommen UP, jetzt wartend: {self.__floor_list[self.current_floor].num_waiting_up()}')
                waiting_up = self.__floor_list[self.current_floor].num_waiting_up()

        if self.__direction == -1 and waiting_down > 0:
            accepted = True
            while self.__num_of_passengers < config.ELEVATOR_PAYLOAD and waiting_down > 0:
                request = yield self.__floor_list[self.current_floor].queue_down.get()
                request.accept_usage_request(self.id)
                self.__num_of_passengers += 1
                log(
                    f'Fahrgast aufgenommen DOWN, jetzt wartend: {self.__floor_list[self.current_floor].num_waiting_down()}')
                waiting_down = self.__floor_list[self.current_floor].num_waiting_down()
        log(f'Fahrgäste im Aufzug: {self.__num_of_passengers}')
        if accepted:
            yield self.__environment.timeout(1)

    def release_passengers(self):
        """
        Checks if any of the passengers inside the elevator have reached their destination.
        If destination of a passenger has been reached their transport event is succeeded and they leave the elevator
        """
        # only do something, if passengers are inside the elevator
        if config.VERBOSE:
            log = print_verbose
        else:
            log = print_silent
        released = False
        if len(self.passenger_requests.items) > 0:
            tmp_q = []
            # Check every passenger inside elevator
            while len(self.passenger_requests.items) > 0:
                request = yield self.passenger_requests.get()
                # release passenger if he is on his desired floor
                if request.destination_floor == self.current_floor:
                    request.reached_floor()
                    self.__num_of_passengers -= 1
                    released = True
                    log(f'Fahrgast auf Etage {self.current_floor} herausgelassen. Anzahl Fahrgäste: {self.__num_of_passengers}')
                    continue
                tmp_q.append(request)

            # re-add passengers still inside the elevator to passenger_requests
            for passenger in tmp_q:
                self.passenger_requests.put(passenger)
        if released:
            yield self.__environment.timeout(1)

    def next_floor(self):
        """
        Moves elevator to the next floor in the currently specified moving direction
        """
        if self.__direction == 1:
            self.current_floor += 1
        if self.__direction == -1:
            self.current_floor -= 1
        yield self.__environment.timeout(1)

    def check_direction_change(self):
        """
        Checks if top or ground floor has been reached and inverts the moving direction of the elevator
        """
        # if ground floor and current direction still set to descend further
        if self.current_floor == 0 and self.__direction == -1:
            # Change direction to ascend
            self.__direction = 1
        # The same if top floor has been reached
        if self.current_floor == config.NUM_OF_FLOORS - 1 and self.__direction == 1:
            # Change direction to descend
            self.__direction = -1


class ElevatorController:
    def __init__(self, environment: simpy.Environment, floor_list: List[Floor], elevator_list: List[Elevator]):
        self.__environment = environment
        self.__floor_list = floor_list
        self.__elevator_list = elevator_list
        self.__environment.process(self.__transport())

    def __transport(self):
        if config.VERBOSE:
            log = print_verbose
        else:
            log = print_silent
        while True:
            # default time between floor
            for elevator in self.__elevator_list:
                elevator.check_direction_change()
                yield from elevator.release_passengers()
                yield from elevator.accept_passengers()
                # wait extra 2 simulation steps if people moved in or out
                yield from elevator.next_floor()
                log("----------------------------------")
            log("==================================")
