import random

from simpy import Environment, Store

# import for intern files
from simulation.util import print_verbose, print_silent
from simulation import config


class Elevator:
    def __init__(self, elevator_id: int,
                 environment: Environment,
                 controller,
                 starting_floor: int):
        self.id = elevator_id
        self.__environment = environment
        self.__controller = controller
        self.current_floor = starting_floor
        self.passenger_requests_store = Store(environment, capacity=config.ELEVATOR_PAYLOAD)
        self.passenger_requests = [0 for _ in range(config.NUM_OF_FLOORS)]
        self.num_of_passengers = 0
        # If the elevator starts on the top floor set initial direction to downwards
        if self.current_floor == config.NUM_OF_FLOORS - 1:
            self.direction = -1
        else:
            self.direction = 1
        self.debug_log = print_silent
        if config.VERBOSE:
            self.debug_log = print_verbose
        # self.__environment.process(self.__transport())

    def __transport(self):
        while True:
            yield self.__environment.process(self.__controller.release_passengers(self))
            yield self.__environment.process(self.__controller.accept_passengers(self))
            yield self.__environment.process(self.__controller.next_floor(self))
            self.debug_log("----------------------------------")


def calc_accept_reward(start_time, end_time):
    time_diff = end_time - start_time
    # return (2/(0.08+np.exp((1/5)*time_diff-5.9))) - 12.5
    # return (2/(0.2+np.exp((1/5)*time_diff-3.5))) / 3
    return 5 / (time_diff + 1)


def calc_release_reward(start_time, end_time):
    time_diff = end_time - start_time
    # return (2/(0.11+np.exp((1/3)*time_diff-6))) - 6
    # return (2 / (0.2 + np.exp((1 / 5) * time_diff - 3.5))) / 3
    return 5 / (time_diff + 1)


class ElevatorController:
    def __init__(self, environment: Environment, skyscraper):
        self.__environment = environment
        self.skyscraper = skyscraper
        self.debug_log = print_silent
        self.step_reward = 0
        if config.VERBOSE:
            self.debug_log = print_verbose

    def next_floor(self, elevator_instance):
        """
        Moves elevator to the next floor in the currently specified moving direction

        :param Elevator elevator_instance:
        """
        elevator_instance.current_floor += elevator_instance.direction
        self.check_direction_change(elevator_instance)
        yield self.__environment.timeout(1)

    def check_direction_change(self, elevator_instance):
        """
        Checks if top or ground floor has been reached and inverts the moving direction of the elevator

        :param Elevator elevator_instance:
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
        Checks if any of the passenger inside the elevator has reached their destination.
        If the destination of a passenger has been reached their transport event is succeeded and they
        leave the elevator

        :param Elevator elevator_instance:
        """
        released = False
        # only do something, if passengers are inside the elevator
        if len(elevator_instance.passenger_requests_store.items) > 0:
            tmp_q = []
            # Check every passenger inside elevator
            while len(elevator_instance.passenger_requests_store.items) > 0:
                request = yield elevator_instance.passenger_requests_store.get()
                # release passenger if he is on his desired floor
                if request.destination_floor == elevator_instance.current_floor:
                    request.reached_floor()
                    elevator_instance.num_of_passengers -= 1
                    elevator_instance.passenger_requests[elevator_instance.current_floor] -= 1
                    released = True
                    self.debug_log(f'Fahrgast auf Etage {elevator_instance.current_floor} herausgelassen. '
                                   f'Anzahl Fahrgäste: {elevator_instance.num_of_passengers}')
                    continue
                tmp_q.append(request)

            # re-add passengers still inside the elevator to passenger_requests
            for passenger in tmp_q:
                elevator_instance.passenger_requests_store.put(passenger)
        # If any passenger left, the elevator had to stop => 1 simulation step
        if released:
            yield self.__environment.timeout(1)

    def accept_passengers(self, elevator_instance):
        """
        Checks if any passenger is waiting on the current floor and accept his usage request if the elevator
        is moving in the correct direction and the maximum capacity hasn't been reached

        :param Elevator elevator_instance:
        """
        accepted = False
        self.debug_log(f'Aufzug ID: {elevator_instance.id}')
        self.debug_log(f'Etage: {elevator_instance.current_floor}')
        self.debug_log(f'Richtung: {elevator_instance.direction}')
        self.debug_log(f'Wartend hoch: {self.skyscraper.floor_list[elevator_instance.current_floor].num_waiting_up}')
        self.debug_log(
            f'Wartend runter: {self.skyscraper.floor_list[elevator_instance.current_floor].num_waiting_down}')

        waiting_up = self.skyscraper.floor_list[elevator_instance.current_floor].num_waiting_up
        waiting_down = self.skyscraper.floor_list[elevator_instance.current_floor].num_waiting_down

        # if the elevator is moving up, there are people waiting on the current floor and the elevator has free capacity
        if elevator_instance.direction == 1 and waiting_up > 0 and elevator_instance.num_of_passengers < config.ELEVATOR_PAYLOAD:
            accepted = True
            # pick up waiting passengers until elevator capacity is reached or no more passenger is waiting
            while elevator_instance.num_of_passengers < config.ELEVATOR_PAYLOAD and waiting_up > 0:
                # accept usage request of the passenger
                request = yield self.skyscraper.floor_list[elevator_instance.current_floor].queue_up.get()
                request.accept_usage_request(elevator_instance.id)
                elevator_instance.num_of_passengers += 1
                self.debug_log(f'Fahrgast aufgenommen, Richtung HOCH, '
                               f'jetzt wartend: {self.skyscraper.floor_list[elevator_instance.current_floor].num_waiting_up}')
                waiting_up = self.skyscraper.floor_list[elevator_instance.current_floor].num_waiting_up

        # if the elevator is moving down, there are people waiting on the current floor and the elevator has free
        # capacity
        if elevator_instance.direction == -1 and waiting_down > 0 and elevator_instance.num_of_passengers < config.ELEVATOR_PAYLOAD:
            accepted = True
            # pick up waiting passengers until elevator capacity has been reached or no more passenger is waiting
            while elevator_instance.num_of_passengers < config.ELEVATOR_PAYLOAD and waiting_down > 0:
                request = yield self.skyscraper.floor_list[elevator_instance.current_floor].queue_down.get()
                request.accept_usage_request(elevator_instance.id)
                elevator_instance.num_of_passengers += 1
                self.debug_log(f'Fahrgast aufgenommen Richtung RUNTER, '
                               f'jetzt wartend: {self.skyscraper.floor_list[elevator_instance.current_floor].num_waiting_down}')
                waiting_down = self.skyscraper.floor_list[elevator_instance.current_floor].num_waiting_down
        self.debug_log(f'Fahrgäste im Aufzug: {elevator_instance.num_of_passengers}')

        # If any passenger was accepted the elevator had to stop => 1 simulation step
        if accepted:
            yield self.__environment.timeout(1)

    # ===============================================================================================================
    # =                                                                                                             =
    # =                                       REF METHODS                                                           =
    # =                                                                                                             =
    # ===============================================================================================================

    def up(self, elevator_instance):
        if elevator_instance.current_floor != config.NUM_OF_FLOORS - 1:
            elevator_instance.current_floor += 1
        yield self.__environment.timeout(1)

    def down(self, elevator_instance):
        if elevator_instance.current_floor != 0:
            elevator_instance.current_floor -= 1
        yield self.__environment.timeout(1)

    def __accept_passenger_and_update_elevator_state(self, elevator_instance):
        waiting_up = self.skyscraper.floor_list[elevator_instance.current_floor].num_waiting_up
        waiting_down = self.skyscraper.floor_list[elevator_instance.current_floor].num_waiting_down
        total_waiting = waiting_up + waiting_down

        hasAccepted = False
        while total_waiting > 0 and elevator_instance.num_of_passengers < config.ELEVATOR_PAYLOAD:
            if waiting_up > 0 and waiting_down > 0:
                hasAccepted = True
                k = random.randint(0, 1)
                if k == 0:
                    request = yield self.skyscraper.floor_list[elevator_instance.current_floor].queue_up.get()
                else:
                    request = yield self.skyscraper.floor_list[elevator_instance.current_floor].queue_down.get()
            elif waiting_up > 0:
                request = yield self.skyscraper.floor_list[elevator_instance.current_floor].queue_up.get()
            else:
                request = yield self.skyscraper.floor_list[elevator_instance.current_floor].queue_down.get()

            # self.skyscraper.step_reward += calc_accept_reward(request.request_time, self.__environment.now)
            request.accept_usage_request(elevator_instance.id)
            elevator_instance.num_of_passengers += 1

            waiting_up = self.skyscraper.floor_list[elevator_instance.current_floor].num_waiting_up
            waiting_down = self.skyscraper.floor_list[elevator_instance.current_floor].num_waiting_down
            total_waiting = waiting_up + waiting_down

        return hasAccepted

    def __release_passenger_and_update_elevator_state(self, elevator_instance):
        released = False
        # only do something, if passengers are inside the elevator
        if len(elevator_instance.passenger_requests_store.items) > 0:
            tmp_q = []
            # Check every passenger inside elevator
            while len(elevator_instance.passenger_requests_store.items) > 0:
                request = yield elevator_instance.passenger_requests_store.get()
                # release passenger if he is on his desired floor
                if request.destination_floor == elevator_instance.current_floor:
                    self.skyscraper.step_reward += calc_release_reward(request.request_time, self.__environment.now)
                    request.reached_floor()
                    elevator_instance.num_of_passengers -= 1
                    elevator_instance.passenger_requests[elevator_instance.current_floor] -= 1
                    released = True
                    self.debug_log(f'Fahrgast auf Etage {elevator_instance.current_floor} herausgelassen. '
                                   f'Anzahl Fahrgäste: {elevator_instance.num_of_passengers}')
                    continue
                tmp_q.append(request)

            # re-add passengers still inside the elevator to passenger_requests
            for passenger in tmp_q:
                elevator_instance.passenger_requests_store.put(passenger)
        return released

    def hold(self, elevator_instance):
        hasReleased = self.__environment.process(self.__release_passenger_and_update_elevator_state(elevator_instance))
        hasAccepted = self.__environment.process(self.__accept_passenger_and_update_elevator_state(elevator_instance))

        yield self.__environment.timeout(1)
