import simpy
import config
import numpy.random as rnd
from passenger import Passenger
from floor import Floor
from elevator import Elevator


class Skyscraper:
    def __init__(self):
        self.__environment = simpy.Environment()
        self.__passenger_list = [0]
        self.__num_of_floors = config.NUM_OF_FLOORS
        self.__num_of_elevators = config.NUM_OF_ELEVATORS
        self.__environment.process(self.__passenger_spawner())
        self.__passenger_id = 0
        self.log = {"waiting_up": [], "waiting_down": []}

        # Create list of available floors (index:0 = ground floor, index:1 = 1. floor, ...)
        self.__floor_list = [Floor(self.__environment, floor_number=x)
                             for x in range(self.__num_of_floors)]
        self.__elevator_list = [Elevator(self.__environment,
                                         starting_floor=x + (14 // self.__num_of_elevators),
                                         floor_list=self.__floor_list)
                                for x in range(self.__num_of_elevators)]

    def __passenger_spawner(self):
        while True:
            waiting_time = rnd.exponential(10)
            yield self.__environment.timeout(waiting_time)

            passenger = Passenger(self.__environment, self.__floor_list, self.__passenger_id)
            print(
                f'{self.__environment.now:.2f} Passenger created: '
                f'Route[{passenger.starting_floor} -> {passenger.destination_floor}]')
            self.update_log_waiting()
            print(self.log)
            self.__passenger_list.append(self.__environment.now)
            self.__passenger_id += 1

    def run_simulation(self, time):
        """
        Run the simulation until the given time is reached

        :param int time: stopping time of the simulation
        """
        self.__environment.run(until=time)

    def update_log_waiting(self):
        """
        Update log of currently waiting passengers for "UP" and "DOWN" for every floor
        """
        waiting_up_per_floor = [x.waiting_up() for x in self.__floor_list]
        waiting_down_per_floor = [x.waiting_down() for x in self.__floor_list]
        self.log["waiting_up"] = waiting_up_per_floor
        self.log["waiting_down"] = waiting_down_per_floor

    """
    def start_simulation(self, until: int = 1000):
        passengers = []
        while self.__environment.peek() < until:
            print(f'{self.__passenger_list[-1]:.2f} neuer Passagier ist erschienen')
            passengers.append(self.__passenger_list[-1])
            self.__environment.step()
        plt.plot(passengers)
        plt.show()
    """


sky = Skyscraper()
sky.run_simulation(1000)
