import simpy
import datetime
import numpy.random as rnd
import matplotlib.pyplot as plt
from typing import List

from simulation import config
from simulation.passenger import Passenger
from simulation.floor import Floor
from simulation.elevator import Elevator, ElevatorController


class Skyscraper:
    def __init__(self, random_seed: int = None):
        self.__environment = simpy.Environment()
        self.__passenger_list = []
        self.__num_of_floors = config.NUM_OF_FLOORS
        self.__num_of_elevators = config.NUM_OF_ELEVATORS
        self.__environment.process(self.__passenger_spawner())
        self.__environment.process(self.__observer())
        self.__elevator_position_log = [[] for _ in range(config.NUM_OF_FLOORS)]
        self.__elevator_utilization_log = [[] for _ in range(config.NUM_OF_ELEVATORS)]
        self.__total_time_log = []
        self.__transportation_time_log = []
        self.__passenger_route_log = []
        self.__log = {"up": [[] for _ in range(config.NUM_OF_FLOORS)],
                      "down": [[] for _ in range(config.NUM_OF_FLOORS)]}
        self.__exp_rate = 100
        # set the random seed to reliably redo a simulation run
        if random_seed is not None:
            rnd.seed(random_seed)

        # Create list of available floors (index:0 = ground floor, index:1 = 1. floor, ...)
        self.__floor_list = [Floor(self.__environment, floor_number=x)
                             for x in range(self.__num_of_floors)]
        # Creates a controller for all available elevator
        self.__elevator_controller = ElevatorController(self.__environment,
                                                        floor_list=self.__floor_list)
        self.__elevator_list = [Elevator(x, self.__environment,
                                         starting_floor=x * 7,
                                         controller=self.__elevator_controller)
                                for x in range(self.__num_of_elevators)]

    @property
    def num_transported_passengers(self):
        """
        Returns the total amount of transported passengers
        """
        return len(self.__total_time_log)

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
            exp_rate, start, destination = self.get_time_dependent_params()
            waiting_time = rnd.exponential(exp_rate)
            yield self.__environment.timeout(waiting_time)
            passenger = Passenger(environment=self.__environment,
                                  floor_list=self.__floor_list,
                                  elevator_list=self.__elevator_list,
                                  time_waited_log=self.__total_time_log,
                                  transportation_time_log=self.__transportation_time_log,
                                  route_log=self.__passenger_route_log,
                                  starting_floor=start,
                                  destination_floor=destination,
                                  passenger_id=passenger_id)
            self.__passenger_list.append(self.__environment.now)
            passenger_id += 1

    def __observer(self):
        while True:
            yield self.__environment.timeout(6)
            self.__log_elevator_position()
            self.__log_waiting_passengers()
            self.__log_elevator_utilization()

    def __log_elevator_position(self):
        for elevator in self.__elevator_list:
            self.__elevator_position_log[elevator.id].append(elevator.current_floor)

    def __log_elevator_utilization(self):
        for elevator in self.__elevator_list:
            self.__elevator_utilization_log[elevator.id].append(elevator.num_of_passengers)

    def __log_waiting_passengers(self):
        """
        Returns the amount of passengers currently waiting to use the elevator (per floor)
        """
        for index, floor in enumerate(self.__floor_list):
            self.__log["up"][index].append(floor.num_waiting_up())
            self.__log["down"][index].append(floor.num_waiting_down())

    def get_time_dependent_params(self) -> List[int]:
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

    def plot_data(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
        fig.set_size_inches(10, 7)
        fig.suptitle('Länge der Warteschlangen', fontsize=16)
        for floor in range(config.NUM_OF_FLOORS):
            ax1.plot(self.__log["up"][floor])
            ax2.plot(self.__log["down"][floor])
        ax1.title.set_text("Warteschlange 'HOCH'")
        ax2.title.set_text("Warteschlange 'RUNTER'")

        exp_rates = []
        r = 1 / config.EXP_RATE_CHECKPOINTS[0][0]
        for x in range(config.SIMULATION_TIME):
            if x in config.EXP_RATE_CHECKPOINTS:
                r = 1 / config.EXP_RATE_CHECKPOINTS[x][0]
            exp_rates.append(r)
        ax3.plot(exp_rates[::6])

        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=None, wspace=None, hspace=0.4)
        plt.show()

        # Displays number of times a floor has appeared as start or destination of a passenger
        num_start = [0 for _ in range(config.NUM_OF_FLOORS)]
        num_destination = [0 for _ in range(config.NUM_OF_FLOORS)]
        for start, destination in self.__passenger_route_log:
            num_start[start] += 1
            num_destination[destination] += 1

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        fig.set_size_inches(10, 7)
        fig.suptitle('Vorkommen der Etagen als Start und Ziel', fontsize=16)
        floor_indices = [x for x in range(len(num_start))]
        bars_start = ax1.bar(floor_indices, num_start, label="Start")
        bars_destination = ax2.bar(floor_indices, num_destination, label="Ziel")
        ax1.bar_label(bars_start)
        ax2.bar_label(bars_destination)
        ax1.title.set_text("Startetage")
        ax2.title.set_text("Zieletage")
        ax1.set_xlabel('Etage')
        ax2.set_xlabel('Etage')
        plt.subplots_adjust(left=0.05, bottom=0.07, right=0.93, top=None, wspace=None, hspace=0.4)
        plt.show()

        # Plot elevator position over time
        plt.rcParams["figure.figsize"] = (13, 7)
        for elevator_id in range(config.NUM_OF_ELEVATORS):
            plt.plot(self.__elevator_position_log[elevator_id], linewidth=0.5)
        plt.show()

        # Plot elevator utilization over time
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
        fig.set_size_inches(10, 7)
        fig.suptitle('Aufzugauslastung', fontsize=16)
        ax1.plot(self.__elevator_utilization_log[0])
        ax2.plot(self.__elevator_utilization_log[1])
        ax3.plot(self.__elevator_utilization_log[2])

        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=None, wspace=None, hspace=0.4)
        plt.show()

    def get_avg_total_time(self) -> float:
        """
        Returns the average time a passenger needed to reach their destination(HH:MM:SS)
        """
        return sum(self.__total_time_log) / len(self.__total_time_log)

    def get_avg_transportation_time(self) -> float:
        """
        Returns the average time a passenger used the elevator
        """
        return sum(self.__transportation_time_log) / len(self.__transportation_time_log)


if __name__ == "__main__":
    sky = Skyscraper(random_seed=12345)
    # time = 8640  // 1 sim step = 10 sec
    sky.run_simulation(8640)
    sky.plot_data()
    avg_total = sky.get_avg_total_time()
    avg_transport = sky.get_avg_transportation_time()
    print('========================================')
    print(f'Anzahl getätigter Anfragen {sky.num_generated_passengers}')
    print(f'Anzahl erfüllter Anfragen: {sky.num_transported_passengers}')
    print(f'Durchn. Zeit zum Ziel: {datetime.timedelta(seconds=avg_total)}')
    print(f'Durchn. Zeit gefahren: {datetime.timedelta(seconds=avg_transport)}')
    print(f'Durchn. Zeit gewartet: {datetime.timedelta(seconds=avg_total - avg_transport)}')
    print('========================================')