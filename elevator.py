import config
import simpy
from typing import List
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

    def accept_passengers(self) -> bool:
        # print(self.__floor_list[self.current_floor].num_waiting_up())
        # as long as there is enough room, accept passengers
        accepted = False
        print(f'Aufzug ID: {self.id}')
        print(f'Etage: {self.current_floor}')
        print(f'Richtung: {self.__direction}')
        print(f'Wartend hoch: {self.__floor_list[self.current_floor].num_waiting_up()}')
        print(f'Wartend runter: {self.__floor_list[self.current_floor].num_waiting_down()}')

        waiting_up = self.__floor_list[self.current_floor].num_waiting_up()
        waiting_down = self.__floor_list[self.current_floor].num_waiting_down()

        if self.__direction == 1 and waiting_up > 0:
            accepted = True
            while self.__num_of_passengers < config.ELEVATOR_PAYLOAD and waiting_up > 0:
                request = yield self.__floor_list[self.current_floor].queue_up.get()
                request.accept_usage_request(self.id)
                self.__num_of_passengers += 1
                print(
                    f'Fahrgast aufgenommen UP, jetzt wartend: {self.__floor_list[self.current_floor].num_waiting_up()}')
                waiting_up = self.__floor_list[self.current_floor].num_waiting_up()

        if self.__direction == -1 and waiting_down > 0:
            accepted = True
            while self.__num_of_passengers < config.ELEVATOR_PAYLOAD and waiting_down > 0:
                request = yield self.__floor_list[self.current_floor].queue_down.get()
                request.accept_usage_request(self.id)
                self.__num_of_passengers += 1
                print(
                    f'Fahrgast aufgenommen DOWN, jetzt wartend: {self.__floor_list[self.current_floor].num_waiting_down()}')
                waiting_down = self.__floor_list[self.current_floor].num_waiting_down()
        print(f'Fahrgäste im Aufzug: {self.__num_of_passengers}')
        return accepted

    def release_passengers(self) -> bool:
        """
        Checks if any of the passengers inside the elevator have reached their destination.
        If destination of a passenger has been reached their transport event is succeeded and they leave the elevator
        """
        # only do something, if passengers are inside the elevator
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
                    print(f'Fahrgast auf Etage {self.current_floor} herausgelassen. Anzahl Fahrgäste: {self.__num_of_passengers}')
                    continue
                tmp_q.append(request)

            # re-add passengers still inside the elevator to passenger_requests
            for passenger in tmp_q:
                self.passenger_requests.put(passenger)
        return released

    def next_floor(self):
        """
        Moves elevator to the next floor in the currently specified moving direction
        """
        if self.__direction == 1:
            self.current_floor += 1
        if self.__direction == -1:
            self.current_floor -= 1

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
        while True:
            # default time between floor
            yield self.__environment.timeout(1)
            for elevator in self.__elevator_list:
                elevator.check_direction_change()
                released = yield from elevator.release_passengers()
                if released:
                    yield self.__environment.timeout(1)
                accepted = yield from elevator.accept_passengers()
                if accepted:
                    yield self.__environment.timeout(1)
                # wait extra 2 simulation steps if people moved in or out
                elevator.next_floor()
                print("----------------------------------")
            print("==================================")
