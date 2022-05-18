import itertools
import simpy
import config
import datetime
import numpy.random as rnd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import plotly.express as px
from passenger import Passenger
from floor import Floor
from elevator import Elevator, ElevatorController


class Skyscraper:
    def __init__(self, random_seed: int= None):
        self.__environment = simpy.Environment()
        self.__passenger_list = []
        self.__num_of_floors = config.NUM_OF_FLOORS
        self.__num_of_elevators = config.NUM_OF_ELEVATORS
        self.__environment.process(self.__passenger_spawner())
        self.__environment.process(self.__floor_observer())
        self.__time_waited_log = []
        self.__log = {"up": [[] for _ in range(config.NUM_OF_FLOORS)], "down": [[] for _ in range(config.NUM_OF_FLOORS)]}
        self.__exp_rate = 100
        # set the random seed to reliably redo a simulation run
        if random_seed is not None:
            rnd.seed(random_seed)

        # Create list of available floors (index:0 = ground floor, index:1 = 1. floor, ...)
        self.__floor_list = [Floor(self.__environment, floor_number=x)
                             for x in range(self.__num_of_floors)]
        self.__elevator_list = [Elevator(x, self.__environment,
                                         starting_floor=x * 7,
                                         floor_list=self.__floor_list)
                                for x in range(self.__num_of_elevators)]
        # Creates a controller for all available elevator
        self.__elevator_controller = ElevatorController(self.__environment,
                                                        floor_list=self.__floor_list,
                                                        elevator_list=self.__elevator_list)

    @property
    def num_transported_passengers(self):
        """
        Returns the total amount of transported passengers
        """
        return len(self.__time_waited_log)

    @property
    def num_generated_passengers(self):
        """
        Returns the total amount of generated passengers
        """
        return len(self.__passenger_list)

    def __passenger_spawner(self):
        passenger_id = 0
        pers = 0
        while True:
            exp_rate, start, destination = self.get_timedependent_params()
            waiting_time = rnd.exponential(exp_rate)
            yield self.__environment.timeout(waiting_time)

            if 2520 <= self.__environment.now < 2880:
                pers += 1
            if 2880 < self.__environment.now <= 2890:
                print(pers)
            passenger = Passenger(environment=self.__environment,
                                  floor_list=self.__floor_list,
                                  elevator_list=self.__elevator_list,
                                  time_waited_log=self.__time_waited_log,
                                  starting_floor=start,
                                  destination_floor=destination,
                                  passenger_id=passenger_id)
            self.__passenger_list.append(self.__environment.now)
            passenger_id += 1

    def __floor_observer(self):
        while True:
            yield self.__environment.timeout(6)
            self.__log_waiting_passengers()

    def __log_waiting_passengers(self):
        """
        Returns the amount of passengers per floor currently waiting to use the elevator
        """
        for index, floor in enumerate(self.__floor_list):
            self.__log["up"][index].append(floor.num_waiting_up())
            self.__log["down"][index].append(floor.num_waiting_down())

    def get_timedependent_params(self) -> List[int]:
        now = int(self.__environment.now)
        possible_exp = config.EXP_RATE_CHECKPOINTS[0]
        for checkpoint in config.EXP_RATE_CHECKPOINTS:
            if now < checkpoint:
                break
            possible_exp = config.EXP_RATE_CHECKPOINTS[checkpoint]
        return possible_exp

    def run_simulation(self, time: int):
        """
        Run the simulation until the given time is reached
        """
        self.__environment.run(until=time)

    def plot_waiting(self, floor: int=None):
        """
        Plots the amount of waiting passengers for the selected floor over time
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        fig.suptitle('Länge Warteschlangen', fontsize=16)
        labels = []
        for floor in range(config.NUM_OF_FLOORS):
            ax1.plot(self.__log["up"][floor])
            ax2.plot(self.__log["down"][floor])
            labels.append(f'Etage {floor}')

        exp_rates = []
        r = 1/config.EXP_RATE_CHECKPOINTS[0][0]
        for x in range(config.SIMULATION_TIME):
            if x in config.EXP_RATE_CHECKPOINTS:
                r = 1/config.EXP_RATE_CHECKPOINTS[x][0]
            exp_rates.append(r)
        ax3.plot(exp_rates[::4])

        plt.show()


    def get_avg_waiting_time(self) -> str:
        """
        Returns the average waiting time of the passengers (HH:MM:SS)
        """
        time_s = sum(self.__time_waited_log) / len(self.__time_waited_log)
        return datetime.timedelta(seconds=time_s).__str__()


if __name__ == "__main__":
    sky = Skyscraper(random_seed=1)
    # time = 8640  // 1 sim step = 10 sec
    sky.run_simulation(8640)
    sky.plot_waiting(floor=0)
    print(f'Anzahl generierter Fahrgäste {sky.num_generated_passengers}')
    print(f'Anzahl transportierter Fahrgäste: {sky.num_transported_passengers}')
    print(f'Durchn. Wartezeit: {sky.get_avg_waiting_time()}')


