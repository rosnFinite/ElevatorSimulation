import simpy
import config
import numpy.random as rnd
import matplotlib.pyplot as plt
from floor import Floor
from elevator import Elevator


class Skyscraper:
    """
    A class representing a Skyscraper

    Attributes
    ----------
    num_of_floors : int
        number of floors
    num_of_elevators : int
        number of usable elevators

    Methods
    -------
    """

    def __init__(self):
        self.__environment = simpy.rt.RealtimeEnvironment(factor=config.SIMULATION_SPEED, strict=False)
        self.__passenger_list = [0]
        self.__num_of_floors = config.NUM_OF_FLOORS
        self.__num_of_elevators = config.NUM_OF_ELEVATORS
        self.__environment.process(self.__passenger_spawner())

        self.floor_list = [Floor(self.__environment, floor_number=x)
                           for x in range(self.__num_of_floors)]
        self.elevator_list = [Elevator(self.__environment,
                                       starting_floor=x + (14 // self.__num_of_elevators),
                                       floor_list=self.floor_list)
                              for x in range(self.__num_of_elevators)]

    def __passenger_spawner(self):
        while True:
            waiting_time = rnd.exponential(10)
            yield self.__environment.timeout(waiting_time)

            self.__passenger_list.append(self.__environment.now)

    def start_simulation(self, until: int = 1000):
        passengers = []
        while self.__environment.peek() < until:
            print(f'{self.__passenger_list[-1]:.2f} neuer Passagier ist erschienen')
            passengers.append(self.__passenger_list[-1])
            self.__environment.step()
        plt.plot(passengers)
        plt.show()


sky = Skyscraper()
sky.start_simulation()
