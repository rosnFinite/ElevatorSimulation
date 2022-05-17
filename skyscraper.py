import simpy
import config
import numpy.random as rnd
import matplotlib.pyplot as plt

from passenger import Passenger
from floor import Floor
from elevator import Elevator, ElevatorController


class Skyscraper:
    def __init__(self):
        self.__environment = simpy.Environment()
        self.__passenger_list = [0]
        self.__num_of_floors = config.NUM_OF_FLOORS
        self.__num_of_elevators = config.NUM_OF_ELEVATORS
        self.__environment.process(self.__passenger_spawner())
        self.__environment.process(self.__floor_observer())
        self.__passenger_id = 0
        self.__log = []

        # Create list of available floors (index:0 = ground floor, index:1 = 1. floor, ...)
        self.__floor_list = [Floor(self.__environment, floor_number=x)
                             for x in range(self.__num_of_floors)]
        self.__elevator_list = [Elevator(x, self.__environment,
                                         starting_floor=x + (14 // self.__num_of_elevators),
                                         floor_list=self.__floor_list)
                                for x in range(self.__num_of_elevators)]
        # Creates a controller for all available elevator
        self.__elevator_controller = ElevatorController(self.__environment,
                                                        floor_list=self.__floor_list,
                                                        elevators=self.__elevator_list)

    def __passenger_spawner(self):
        while True:
            waiting_time = rnd.exponential(10)
            yield self.__environment.timeout(waiting_time)

            passenger = Passenger(self.__environment, self.__floor_list, self.__elevator_list, self.__passenger_id)
            """
            print(
                f'{self.__environment.now:.2f} Passenger created: '
                f'Route[{passenger.starting_floor} -> {passenger.destination_floor}]')
            """
            self.__passenger_list.append(self.__environment.now)
            self.__passenger_id += 1

    def __floor_observer(self):
        while True:
            yield self.__environment.timeout(4)
            self.__log.append(self.__get_waiting_passengers())

    def __get_waiting_passengers(self):
        """
        Returns the amount of passengers per floor currently waiting to use the elevator

        :return Dict[str, int]:
        """
        tmp_log = {"waiting_up": [], "waiting_down":[]}
        waiting_up_per_floor = [x.num_waiting_up() for x in self.__floor_list]
        waiting_down_per_floor = [x.num_waiting_down() for x in self.__floor_list]
        tmp_log["waiting_up"] = waiting_up_per_floor
        tmp_log["waiting_down"] = waiting_down_per_floor
        return tmp_log

    def run_simulation(self, time):
        """
        Run the simulation until the given time is reached

        :param int time: stopping time of the simulation
        """
        self.__environment.run(until=time)

    def plot_waiting(self, floor):
        """
        Plots the amount of waiting passengers for the selected floor over time

        :param int floor: Floor between 0 and 14
        :return:
        """
        waiting = []
        for value in self.__log:
            waiting.append(value["waiting_up"][floor])
        plt.plot(waiting)
        plt.show()



if __name__ == "__main__":
    sky = Skyscraper()
    sky.run_simulation(5760)
    sky.plot_waiting(floor=0)


