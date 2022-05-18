import simpy
import config
from typing import List
from util import print_verbose, print_silent
from floor import Floor


class Elevator:
    def __init__(self, elevator_id: int,
                 environment: simpy.Environment,
                 controller,
                 starting_floor: int):
        self.id = elevator_id
        self.__environment = environment
        self.__controller = controller
        self.current_floor = starting_floor
        self.passenger_requests = simpy.Store(environment, capacity=config.ELEVATOR_PAYLOAD)
        self.num_of_passengers = 0
        if self.current_floor == config.NUM_OF_FLOORS-1:
            self.direction = -1
        else:
            self.direction = 1

        self.__environment.process(self.__transport())

    def __transport(self):
        if config.VERBOSE:
            log = print_verbose
        else:
            log = print_silent
        while True:
            yield self.__environment.process(self.__controller.release_passengers(self))
            yield self.__environment.process(self.__controller.accept_passengers(self))
            yield self.__environment.process(self.__controller.next_floor(self))
            log("----------------------------------")


class ElevatorController:
    def __init__(self, environment: simpy.Environment, floor_list: List[Floor]):
        self.__environment = environment
        self.__floor_list = floor_list

    def next_floor(self, elevator_instance):
        """
        Moves elevator to the next floor in the currently specified moving direction
        """
        if elevator_instance.direction == 1:
            elevator_instance.current_floor += 1
        if elevator_instance.direction == -1:
            elevator_instance.current_floor -= 1
        self.check_direction_change(elevator_instance)
        yield self.__environment.timeout(1)

    def check_direction_change(self, elevator_instance):
        """
        Checks if top or ground floor has been reached and inverts the moving direction of the elevator
        """
        # if ground floor and current direction still set to descend further
        if elevator_instance.current_floor == 0 and elevator_instance.direction == -1:
            # Change direction to ascend
            elevator_instance.direction = 1
        # The same if top floor has been reached
        if elevator_instance.current_floor == config.NUM_OF_FLOORS - 1 and elevator_instance.direction == 1:
            # Change direction to descend
            elevator_instance.direction = -1

    def release_passengers(self, elevator_instance):
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
        if len(elevator_instance.passenger_requests.items) > 0:
            tmp_q = []
            # Check every passenger inside elevator
            while len(elevator_instance.passenger_requests.items) > 0:
                request = yield elevator_instance.passenger_requests.get()
                # release passenger if he is on his desired floor
                if request.destination_floor == elevator_instance.current_floor:
                    request.reached_floor()
                    elevator_instance.num_of_passengers -= 1
                    released = True
                    log(f'Fahrgast auf Etage {elevator_instance.current_floor} herausgelassen. Anzahl Fahrgäste: {elevator_instance.num_of_passengers}')
                    continue
                tmp_q.append(request)

            # re-add passengers still inside the elevator to passenger_requests
            for passenger in tmp_q:
                elevator_instance.passenger_requests.put(passenger)
        if released:
            yield self.__environment.timeout(1)

    def accept_passengers(self, elevator_instance):
        # print(self.__floor_list[self.current_floor].num_waiting_up())
        # as long as there is enough room, accept passengers
        if config.VERBOSE:
            log = print_verbose
        else:
            log = print_silent
        accepted = False
        log(f'Aufzug ID: {elevator_instance.id}')
        log(f'Etage: {elevator_instance.current_floor}')
        log(f'Richtung: {elevator_instance.direction}')
        log(f'Wartend hoch: {self.__floor_list[elevator_instance.current_floor].num_waiting_up()}')
        log(f'Wartend runter: {self.__floor_list[elevator_instance.current_floor].num_waiting_down()}')

        waiting_up = self.__floor_list[elevator_instance.current_floor].num_waiting_up()
        waiting_down = self.__floor_list[elevator_instance.current_floor].num_waiting_down()

        if elevator_instance.direction == 1 and waiting_up > 0:
            accepted = True
            while elevator_instance.num_of_passengers < config.ELEVATOR_PAYLOAD and waiting_up > 0:
                request = yield self.__floor_list[elevator_instance.current_floor].queue_up.get()
                request.accept_usage_request(elevator_instance.id)
                elevator_instance.num_of_passengers += 1
                log(
                    f'Fahrgast aufgenommen UP, jetzt wartend: {self.__floor_list[elevator_instance.current_floor].num_waiting_up()}')
                waiting_up = self.__floor_list[elevator_instance.current_floor].num_waiting_up()

        if elevator_instance.direction == -1 and waiting_down > 0:
            accepted = True
            while elevator_instance.num_of_passengers < config.ELEVATOR_PAYLOAD and waiting_down > 0:
                request = yield self.__floor_list[elevator_instance.current_floor].queue_down.get()
                request.accept_usage_request(elevator_instance.id)
                elevator_instance.num_of_passengers += 1
                log(
                    f'Fahrgast aufgenommen DOWN, jetzt wartend: {self.__floor_list[elevator_instance.current_floor].num_waiting_down()}')
                waiting_down = self.__floor_list[elevator_instance.current_floor].num_waiting_down()
        log(f'Fahrgäste im Aufzug: {elevator_instance.num_of_passengers}')
        if accepted:
            yield self.__environment.timeout(1)
