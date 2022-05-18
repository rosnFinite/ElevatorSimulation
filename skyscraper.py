import simpy
import config
import datetime
import numpy.random as rnd
import matplotlib.pyplot as plt
from typing import List
from bokeh.plotting import figure, show
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
        self.__log = []
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
        while True:
            exp_rate, start, destination = self.get_timedependent_params()
            waiting_time = rnd.exponential(exp_rate)
            yield self.__environment.timeout(waiting_time)

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
            self.__log.append(self.__get_waiting_passengers())

    def __get_waiting_passengers(self) -> dict[str, int]:
        """
        Returns the amount of passengers per floor currently waiting to use the elevator
        """
        tmp_log = {"waiting_up": [], "waiting_down":[]}
        waiting_up_per_floor = [x.num_waiting_up() for x in self.__floor_list]
        waiting_down_per_floor = [x.num_waiting_down() for x in self.__floor_list]
        tmp_log["waiting_up"] = waiting_up_per_floor
        tmp_log["waiting_down"] = waiting_down_per_floor
        return tmp_log

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

    def plot_waiting(self, floor: int):
        """
        Plots the amount of waiting passengers for the selected floor over time

        waiting = []
        for value in self.__log:
            waiting.append(value["waiting_up"][floor])
        plt.plot(waiting)
        plt.show()
        """
        waiting = []
        for value in self.__log:
            waiting.append(value["waiting_up"][floor])
        p = figure(title=f'Wartende Fahrgäste Etage {floor}', x_axis_label='x', y_axis_label='y')
        p.line([x for x in range(len(waiting))], waiting, legend_label="Anzahl wartender Personen", line_width=1)
        show(p)

    def get_avg_waiting_time(self) -> str:
        """
        Returns the average waiting time of the passengers (HH:MM:SS)
        """
        time_s = sum(self.__time_waited_log) / len(self.__time_waited_log)
        return datetime.timedelta(seconds=time_s).__str__()


if __name__ == "__main__":
    sky = Skyscraper(random_seed=1234567)
    # time = 8640  // 1 sim step = 10 sec
    sky.run_simulation(8640)
    sky.plot_waiting(floor=0)
    print(f'Anzahl generierter Fahrgäste {sky.num_generated_passengers}')
    print(f'Anzahl transportierter Fahrgäste: {sky.num_transported_passengers}')
    print(f'Durchn. Wartezeit: {sky.get_avg_waiting_time()}')


