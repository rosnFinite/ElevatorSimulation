import enum
import time

import simpy
import datetime
import numpy.random as rnd
import pandas as pd
import numpy as np
from typing import List

from simulation import config
from simulation.passenger import Passenger
from simulation.floor import Floor
from simulation.elevator import Elevator, ElevatorController


class Skyscraper:
    def __init__(self, random_seed: int = None, passenger_behaviour: dict = None, is_scanning=False):
        self.environment = simpy.Environment()
        self.environment.process(self.__passenger_spawner())
        self.environment.process(self.__observer())
        self.num_of_floors = config.NUM_OF_FLOORS
        self.num_of_elevators = config.NUM_OF_ELEVATORS
        self.step_reward = 0
        self.is_scanning = False
        # --------------------------Simulation logs--------------------------
        self.num_passengers = 0
        self.elevator_position_log = [[] for _ in range(config.NUM_OF_ELEVATORS)]
        self.elevator_utilization_log = [[] for _ in range(config.NUM_OF_ELEVATORS)]
        self.total_time_log = []
        self.queue_time_log = []
        self.travel_time_log = []
        self.passenger_route_log = []
        self.queue_usage_log = {"up": [[] for _ in range(config.NUM_OF_FLOORS)],
                                "down": [[] for _ in range(config.NUM_OF_FLOORS)]}
        # combined dataframe of position/utilization/q_up/q_down for better usage with plotly
        # created after simulation is finished
        self.df_log = None
        # -------------------------------------------------------------------
        # set a random seed to reliably redo a simulation run
        if random_seed is not None:
            rnd.seed(random_seed)
        self.passenger_behaviour = config.PASSENGER_BEHAVIOUR
        # if custom passenger behaviour is passed, use it
        if passenger_behaviour is not None:
            self.passenger_behaviour = passenger_behaviour

        # Create list of available floors (0 = ground floor, 1 = 1. floor, ..., 14 = 14. floor)
        self.floor_list = [Floor(self.environment, floor_number=x)
                           for x in range(self.num_of_floors)]
        # Creates a controller for all available elevator
        self.elevator_controller = ElevatorController(self.environment, self)
        self.elevator_list = [Elevator(x, self.environment,
                                       starting_floor=x * 7,
                                       is_scanning=is_scanning,
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
        return self.num_passengers

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
        while True:
            # get the current behaviour parameters
            exp_rate, start_f, destination_f = self.__get_time_dependent_params()
            waiting_time = rnd.exponential(exp_rate)
            yield self.environment.timeout(waiting_time)
            Passenger(environment=self.environment,
                      skyscraper=self,
                      starting_floor=start_f,
                      destination_floor=destination_f,
                      passenger_id=self.num_passengers)
            self.num_passengers += 1

    def __observer(self):
        while True:
            yield self.environment.timeout(1)
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
        """
        Compares the current simulation step with the  checkpoints given in passenger_behaviour.
        If a checkpoint is reached the defined expectancy value and spawnposition for that checkpoint will be used
        """
        now = int(self.environment.now)
        dependent_params = self.passenger_behaviour[0]
        for checkpoint in self.passenger_behaviour:
            if now < checkpoint:
                break
            dependent_params = self.passenger_behaviour[checkpoint]
        return dependent_params

    def __create_df_log(self):
        """
        Converts the separate logs of position/utilization/q_up/q_down into one single dataframe
        for better integration with plotly.
        """
        prep_q_up = {f'up_{x}': self.get_queue_up_log[x] for x in range(config.NUM_OF_FLOORS)}
        prep_q_down = {f'down_{x}': self.get_queue_down_log[x] for x in range(config.NUM_OF_FLOORS)}
        prep_el_pos = dict(e0_pos=self.elevator_position_log[0], e1_pos=self.elevator_position_log[1],
                           e2_pos=self.elevator_position_log[2])
        prep_el_util = dict(e0_util=self.elevator_utilization_log[0], e1_util=self.elevator_utilization_log[1],
                            e2_util=self.elevator_utilization_log[2])
        df_q_up = pd.DataFrame.from_dict(prep_q_up)
        df_q_down = pd.DataFrame.from_dict(prep_q_down)
        df_el_pos = pd.DataFrame.from_dict(prep_el_pos)
        df_el_util = pd.DataFrame.from_dict(prep_el_util)
        self.df_log = pd.concat([df_q_up, df_q_down, df_el_pos, df_el_util], axis=1)

    def run_simulation(self, until_time: int):
        """
        Run the simulation until the given time is reached
        """
        self.environment.run(until=until_time)
        self.__create_df_log()

    def schedule_action(self, action_list):
        """
        Start processes responsible for performing Agents action.
        """
        for index, a in enumerate(action_list):
            if a == 0:
                self.environment.process(self.elevator_controller.up(self.elevator_list[index]))
            if a == 1:
                self.environment.process(self.elevator_controller.down(self.elevator_list[index]))
            if a == 2:
                self.environment.process(self.elevator_controller.hold(self.elevator_list[index]))

    def step(self):
        """
        Runs simulation for one time step. Returns next state
        """
        self.environment.run(until=self.environment.now+1)

        return self.get_state()

    def get_state(self):
        max_up = max([floor.num_waiting_up for floor in self.floor_list]) + 10e-13
        max_down = max([floor.num_waiting_down for floor in self.floor_list]) + 10e-13
        f_waiting_up = [floor.num_waiting_up/max_up for floor in self.floor_list]  # 15
        f_waiting_down = [floor.num_waiting_down/max_down for floor in self.floor_list]  # 15

        e0_pos = [0 if x != self.elevator_list[0].current_floor else 1 for x in range(config.NUM_OF_FLOORS)]  # 15
        e1_pos = [0 if x != self.elevator_list[1].current_floor else 1 for x in range(config.NUM_OF_FLOORS)]  # 15
        e2_pos = [0 if x != self.elevator_list[2].current_floor else 1 for x in range(config.NUM_OF_FLOORS)]  # 15

        e0_util = self.elevator_list[0].num_of_passengers / config.ELEVATOR_PAYLOAD  # 1
        e1_util = self.elevator_list[1].num_of_passengers / config.ELEVATOR_PAYLOAD  # 1
        e2_util = self.elevator_list[2].num_of_passengers / config.ELEVATOR_PAYLOAD  # 1

        e0_destinations = [x/(config.NUM_OF_FLOORS-1) for x in self.elevator_list[0].passenger_requests]  # 15
        e1_destinations = [x/(config.NUM_OF_FLOORS-1) for x in self.elevator_list[1].passenger_requests]
        e2_destinations = [x/(config.NUM_OF_FLOORS-1) for x in self.elevator_list[2].passenger_requests]
        sim_time = self.environment.now / (config.SIMULATION_TIME-1)
        state = ([sim_time] + f_waiting_up + f_waiting_down + e0_pos + e1_pos + e2_pos +
                 [e0_util] + [e1_util] + [e2_util] + e0_destinations + e1_destinations + e2_destinations)
        reward = self.step_reward
        self.step_reward = 0
        isDone = False

        if self.environment.now == config.SIMULATION_TIME-1:
            isDone = True
            # create final dataframe for dash visualization
            self.__create_df_log()

        return state, reward, isDone

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
    # sky.run_simulation(config.SIMULATION_TIME)
    print(sky.step())

