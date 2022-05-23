import simpy
import datetime
import numpy.random as rnd
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from simulation import config
from simulation.passenger import Passenger
from simulation.floor import Floor
from simulation.elevator import Elevator, ElevatorController


class Skyscraper:
    def __init__(self, random_seed: int = None, passenger_behaviour: dict = None):
        self.__environment = simpy.Environment()
        self.__environment.process(self.__passenger_spawner())
        self.__environment.process(self.__observer())
        self.passenger_list = []
        self.num_of_floors = config.NUM_OF_FLOORS
        self.num_of_elevators = config.NUM_OF_ELEVATORS
        # --------------------------Simulation logs--------------------------
        self.elevator_position_log = [[] for _ in range(config.NUM_OF_FLOORS)]
        self.elevator_utilization_log = [[] for _ in range(config.NUM_OF_ELEVATORS)]
        self.total_time_log = []
        self.queue_time_log = []
        self.travel_time_log = []
        self.passenger_route_log = []
        self.queue_usage_log = {"up": [[] for _ in range(config.NUM_OF_FLOORS)],
                                "down": [[] for _ in range(config.NUM_OF_FLOORS)]}
        # set the random seed to reliably redo a simulation run
        if random_seed is not None:
            rnd.seed(random_seed)
        self.passenger_behaviour = config.PASSENGER_BEHAVIOUR
        # if custom passenger behaviour is passed, use it
        if passenger_behaviour is not None:
            self.passenger_behaviour = passenger_behaviour

        # Create list of available floors (index:0 = ground floor, index:1 = 1. floor, ...)
        self.floor_list = [Floor(self.__environment, floor_number=x)
                           for x in range(self.num_of_floors)]
        # Creates a controller for all available elevator
        self.elevator_controller = ElevatorController(self.__environment, self)
        self.elevator_list = [Elevator(x, self.__environment,
                                       starting_floor=x * 7,
                                       controller=self.elevator_controller)
                              for x in range(self.num_of_elevators)]

    @property
    def num_transported_passengers(self):
        """
        Returns the total amount of transported passengers
        """
        return len(self.total_time_log)

    @property
    def num_generated_passengers(self):
        """
        Returns the total amount of generated passengers
        """
        return len(self.passenger_list)

    @property
    def mean_queue_time(self):
        """
        Returns the average time to wait for an elevator
        """
        return np.mean(self.queue_time_log)

    @property
    def median_queue_time(self):
        """
        Returns the median time to wait for an elevator
        """
        return np.median(self.queue_time_log)

    @property
    def std_queue_time(self):
        """
        Returns the standard deviation on every logged queue time
        """
        return np.std(self.queue_time_log)

    @property
    def mean_travel_time(self):
        """
        Returns the average time to reach destination after entering the elevator
        """
        return np.mean(self.travel_time_log)

    @property
    def median_travel_time(self):
        """
        Returns the median travel time
        """
        return np.median(self.travel_time_log)

    @property
    def std_travel_time(self):
        """
        Returns the standard deviation on every logged travel time
        """
        return np.std(self.travel_time_log)

    @property
    def get_queue_up_log(self):
        return [self.queue_usage_log["up"][x] for x in range(config.NUM_OF_FLOORS)]

    @property
    def get_queue_down_log(self):
        return [self.queue_usage_log["down"][x] for x in range(config.NUM_OF_FLOORS)]

    def __passenger_spawner(self):
        passenger_id = 0
        while True:
            exp_rate, start, destination = self.__get_time_dependent_params()
            waiting_time = rnd.exponential(exp_rate)
            yield self.__environment.timeout(waiting_time)
            Passenger(environment=self.__environment,
                      skyscraper=self,
                      starting_floor=start,
                      destination_floor=destination,
                      passenger_id=passenger_id)
            self.passenger_list.append(self.__environment.now)
            passenger_id += 1

    def __observer(self):
        while True:
            yield self.__environment.timeout(6)
            self.__log_elevator_position()
            self.__log_waiting_passengers()
            self.__log_elevator_utilization()

    def __log_elevator_position(self):
        for elevator in self.elevator_list:
            self.elevator_position_log[elevator.id].append(elevator.current_floor)

    def __log_elevator_utilization(self):
        for elevator in self.elevator_list:
            self.elevator_utilization_log[elevator.id].append(elevator.num_of_passengers)

    def __log_waiting_passengers(self):
        """
        Logs the amount of passengers currently waiting to use the elevator (per floor)
        """
        for index, floor in enumerate(self.floor_list):
            self.queue_usage_log["up"][index].append(floor.num_waiting_up)
            self.queue_usage_log["down"][index].append(floor.num_waiting_down)

    def __get_time_dependent_params(self) -> List[int]:
        now = int(self.__environment.now)
        dependent_params = self.passenger_behaviour[0]
        for checkpoint in self.passenger_behaviour:
            if now < checkpoint:
                break
            dependent_params = self.passenger_behaviour[checkpoint]
        return dependent_params

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
            ax1.plot(self.queue_usage_log["up"][floor])
            ax2.plot(self.queue_usage_log["down"][floor])
        ax1.title.set_text("Warteschlange 'HOCH'")
        ax2.title.set_text("Warteschlange 'RUNTER'")

        exp_rates = []
        r = 1 / self.passenger_behaviour[0][0]
        for x in range(config.SIMULATION_TIME):
            if x in self.passenger_behaviour:
                r = 1 / self.passenger_behaviour[x][0]
            exp_rates.append(r)
        ax3.plot(exp_rates[::6])

        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=None, wspace=None, hspace=0.4)
        plt.show()

        # Displays number of times a floor has appeared as start or destination of a passenger
        num_start = [0 for _ in range(config.NUM_OF_FLOORS)]
        num_destination = [0 for _ in range(config.NUM_OF_FLOORS)]
        for start, destination in self.passenger_route_log:
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
            plt.plot(self.elevator_position_log[elevator_id], linewidth=0.5)
        plt.show()

        # Plot elevator utilization over time
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
        fig.set_size_inches(10, 7)
        fig.suptitle('Aufzugauslastung', fontsize=16)
        ax1.plot(self.elevator_utilization_log[0])
        ax2.plot(self.elevator_utilization_log[1])
        ax3.plot(self.elevator_utilization_log[2])

        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=None, wspace=None, hspace=0.4)
        plt.show()

    def statistics(self):
        avg_total = self.mean_travel_time + self.mean_queue_time
        avg_travel = self.mean_travel_time
        return '========================================\n' \
               f'Anzahl getätigter Anfragen {self.num_generated_passengers}\n' \
               f'Anzahl erfüllter Anfragen: {self.num_transported_passengers}\n' \
               f'Durchn. Zeit zum Ziel: {datetime.timedelta(seconds=avg_total)}\n' \
               f'Durchn. Zeit gefahren: {datetime.timedelta(seconds=avg_travel)}\n' \
               f'Durchn. Zeit gewartet: {datetime.timedelta(seconds=avg_total - avg_travel)}\n' \
               f'========================================'


if __name__ == "__main__":
    sky = Skyscraper(random_seed=12345)
    # time = 8640  // 1 sim step = 10 sec
    sky.run_simulation(config.SIMULATION_TIME)
    sky.plot_data()
    print(sky.statistics())
