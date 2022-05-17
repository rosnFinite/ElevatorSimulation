import config
import simpy
from typing import List
from floor import Floor


class ElevatorController:
    def __init__(self, environment, floor_list, elevators):
        self.__environment = environment
        self.__floor_list = floor_list
        self.__elevator_list = elevators
        self.__environment.process(self.__transport())

    def __transport(self):
        while True:
            yield self.__environment.timeout(100)
            for elevator in self.__elevator_list:
                elevator.check_direction_change()
                # elevator.release_passengers()
                yield from elevator.accept_passengers()
                print("----------------------------------")


class Elevator:
    def __init__(self, id, environment, floor_list, starting_floor):
        """
        :param int id:
        :param simpy.Environment environment:
        :param List[Floor] floor_list:
        :param int starting_floor:
        """
        self.id = id
        self.__environment = environment
        self.current_floor = starting_floor
        self.__floor_list = floor_list
        self.passenger_requests = simpy.Store(environment, capacity=config.ELEVATOR_PAYLOAD)
        self.__num_of_passengers = 0
        self.__direction = 1

    def accept_passengers(self):
        # print(self.__floor_list[self.current_floor].num_waiting_up())
        # as long as there is enough room, accept passengers
        print(f'Aufzug ID: {self.id}')
        print(f'Richtung: {self.__direction}')
        print(f'Wartend hoch: {self.__floor_list[self.current_floor].num_waiting_up()}')
        print(f'Wartend runter: {self.__floor_list[self.current_floor].num_waiting_down()}')

        waiting_up = self.__floor_list[self.current_floor].num_waiting_up()
        waiting_down = self.__floor_list[self.current_floor].num_waiting_down()

        # while elevator has free capacity and people are waiting, accept them
        while self.__num_of_passengers < config.ELEVATOR_PAYLOAD and (waiting_up > 0 and waiting_down > 0):
            # if direction is up, and passengers are waiting for up
            if self.__direction == 1 and self.__floor_list[self.current_floor].num_waiting_up() > 0:
                # accept their usage request
                request = yield self.__floor_list[self.current_floor].queue_up.get()
                request.accept_usage_request(self.id)
                self.__num_of_passengers += 1
                print(
                    f'Fahrgast aufgenommen UP, jetzt wartend: {self.__floor_list[self.current_floor].num_waiting_up()}')
            # same for direction down
            elif self.__direction == -1 and self.__floor_list[self.current_floor].num_waiting_down() > 0:
                request = yield self.__floor_list[self.current_floor].queue_down.get()
                request.accept_usage_request(self.id)
                self.__num_of_passengers += 1
                print(
                    f'Fahrgast aufgenommen DOWN, jetzt wartend: {self.__floor_list[self.current_floor].num_waiting_down()}')

            # update amount of people waiting
            waiting_up = self.__floor_list[self.current_floor].num_waiting_up()
            waiting_down = self.__floor_list[self.current_floor].num_waiting_down()

        print(f'Fahrgäste im Aufzug: {self.__num_of_passengers}')

        """
        while len(self.passenger_requests.items) < config.ELEVATOR_PAYLOAD:
            if self.__direction == 1:
                if self.__floor_list[self.current_floor].num_waiting_up() > 0:
                    print("up")
                    
                    request = yield self.__floor_list[self.current_floor].queue_up.get()
                    print(request)
                    request.fullfill_usage_request(self.id)
                else:
                    break
            if self.__direction == -1:
                if self.__floor_list[self.current_floor].num_waiting_down() > 0:
                    request = yield self.__floor_list[self.current_floor].queue_down.get()
                    request.fullfill_usage_request(self.id)
                else:
                    break
                    """

    def release_passengers(self):
        # only do something, if passengers are inside the elevator
        if len(self.passenger_requests.items) > 0:
            tmp_q = []
            # Check every passenger inside elevator
            while len(self.passenger_requests.items) > 0:
                request = self.passenger_requests.get()
                # release passenger if he is on his desired floor
                # TODO request has no attribute destination_floor
                if request.destination_floor == self.current_floor:
                    request.reached_floor()
                    self.__num_of_passengers -= 1
                    continue
                tmp_q.append(request)

            # re-add passengers still inside the elevator to passenger_requests
            for passenger in tmp_q:
                self.passenger_requests.put(passenger)

    def check_direction_change(self):
        """
        Checks if top or bottom floor has been reached and changes direction of the elevator
        :return:
        """
        # if ground floor and current direction still set to descend further
        if self.current_floor == 0 and self.__direction == -1:
            # Change direction to ascend
            self.__direction = 1
        # The same if top floor has been reached
        if self.current_floor == config.NUM_OF_FLOORS - 1 and self.__direction == 1:
            # Change direction to descend
            self.__direction = -1
