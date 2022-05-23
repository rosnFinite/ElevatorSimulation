import sys
import dash_mantine_components as dmc
import datetime
from dash import Dash, Output, Input, State
from dash_iconify import DashIconify

# intern functions
import plotly.graph_objects as go
import utils.plotting as plotting
from utils.parameter_preparation import create_passenger_behaviour
from components.header import MainHeader
from components.menu import EnvironmentParameterMenu, PassengerBehaviourMenu
from components.visuals import PassengerSpawnratePlot
from components.statistics import TextualStats
from simulation.skyscraper import Skyscraper

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]
this.passenger_behaviour = None

app = Dash(__name__)

app.layout = dmc.Container(
    children=[
        MainHeader,
        dmc.Accordion(
            multiple=True,
            disableIconRotation=True,
            state={"0": True},
            children=[
                dmc.AccordionItem(
                    label="Run Simulation",
                    icon=[
                        DashIconify(
                            icon="el:play",
                            color=dmc.theme.DEFAULT_COLORS["green"][6],
                            width=20,
                        )
                    ],
                    children=[
                        dmc.Grid(
                            children=[
                                dmc.Col(EnvironmentParameterMenu, span=3),
                                dmc.Col(PassengerBehaviourMenu, span=3),
                                dmc.Col(PassengerSpawnratePlot, span=6)
                            ],
                            style={"marginTop": 20},
                            gutter="xs",
                            grow=True
                        ),
                        dmc.Center(
                            children=[
                                dmc.Button(
                                    "Start Simulation",
                                    id="simulation-button",
                                    leftIcon=[DashIconify(icon="el:play")],
                                    variant="gradient",
                                    gradient={"from": "teal", "to": "lime", "deg": 105},
                                    size="md"
                                ),
                            ],
                            style={"width": "100%", "marginTop": 20}
                        )
                    ]
                ),
                dmc.AccordionItem(
                    label="Statistics",
                    icon=[
                        DashIconify(
                            icon="carbon:chart-line-data",
                            color=dmc.theme.DEFAULT_COLORS["green"][6],
                            width=20,
                        )
                    ],
                    children=[
                        TextualStats
                    ]
                ),
                dmc.AccordionItem(
                    label="Visualization",
                    icon=[
                        DashIconify(
                            icon="simple-icons:plotly",
                            color=dmc.theme.DEFAULT_COLORS["green"][6],
                            width=20,
                        )
                    ],
                    children=[

                    ]
                )
            ]
        )
    ],
    fluid=True
)


# SimulationParameterMenu
@app.callback(
    Output(component_id="input-simulation-steps", component_property="description"),
    Input(component_id="input-simulation-steps", component_property="value")
)
def update_total_sim_steps(seconds_per_step):
    return f'Total simulation steps: {int(1440*(60/seconds_per_step))}'


# -----------------------------RUN SIMULATION AND DISPLAY STATS-----------------------------
@app.callback(
    [
        Output(component_id="text-total-spawned", component_property="children"),
        Output(component_id="text-total-transported", component_property="children"),
        Output(component_id="text-total-abandoned", component_property="children"),
        Output(component_id="text-percentage-transported", component_property="children"),
        Output(component_id="text-mean-queue-time", component_property="children"),
        Output(component_id="text-median-queue-time", component_property="children"),
        Output(component_id="text-deviation-queue-time", component_property="children"),
        Output(component_id="text-mean-elevator-time", component_property="children"),
        Output(component_id="text-median-elevator-time", component_property="children"),
        Output(component_id="text-deviation-elevator-time", component_property="children"),
        Output(component_id="queue-up-plot", component_property="figure"),
        Output(component_id="queue-down-plot", component_property="figure")
    ],
    [
        State(component_id="input-simulation-steps", component_property="value"),
        State(component_id="input-random-seed", component_property="value")
    ],
    Input("simulation-button", "n_clicks"),
    prevent_initial_call=True
)
def get_simulation_data(seconds_per_step, random_seed, n_clicks):
    if random_seed is not None:
        sky = Skyscraper(random_seed, passenger_behaviour=this.passenger_behaviour)
    else:
        sky = Skyscraper(passenger_behaviour=this.passenger_behaviour)
    sky.run_simulation(time=int(1440*(60/seconds_per_step)))
    total_spawned = f'#Passengers created: {sky.num_generated_passengers}'
    total_transported = f'#Passengers transported: {sky.num_transported_passengers}'
    total_abandoned = f'#Passengers abandoned: {sky.num_generated_passengers - sky.num_transported_passengers}'
    percentage_transported = f'Quota: {sky.num_transported_passengers/sky.num_generated_passengers * 100:.2f}%'
    mean_queue_time = f'Mean: {datetime.timedelta(seconds=sky.mean_queue_time)}'
    median_queue_time = f'Median: {datetime.timedelta(seconds=sky.median_queue_time)}'
    std_queue_time = f'Std. Deviation: {datetime.timedelta(seconds=sky.std_queue_time)}'
    mean_elevator_time = f'Mean: {datetime.timedelta(seconds=sky.mean_travel_time)}'
    median_elevator_time = f'Median: {datetime.timedelta(seconds=sky.mean_travel_time)}'
    std_elevator_time = f'Std. Deviation: {datetime.timedelta(seconds=sky.std_travel_time)}'
    fig_up = plotting.plot_waiting(sky, "up")
    fig_down = plotting.plot_waiting(sky, "down")

    return total_spawned, total_transported, total_abandoned, percentage_transported, mean_queue_time, \
        median_queue_time, std_queue_time, mean_elevator_time, median_elevator_time, std_elevator_time, fig_up, \
        fig_down


# -----------------------------UPDATE PASSENGER SPAWN RATE PLOT-----------------------------
@app.callback(
    Output(component_id="spawn-plot", component_property="figure"),
    Input(component_id="passenger-spawn-radiogroup", component_property="value"),
    Input(component_id="passenger-floor-radiogroup", component_property="value"),
    Input(component_id="input-simulation-steps", component_property="value"),
)
def update_spawn_behaviour_visual(spawn_behaviour, floor_behaviour, seconds_per_step):
    this.passenger_behaviour = create_passenger_behaviour(seconds_per_step, spawn_behaviour, floor_behaviour)
    sim_steps = int(1440*(60/seconds_per_step))
    exp_rates = []
    r = 1 / this.passenger_behaviour[0][0]
    for x in range(sim_steps):
        if x in this.passenger_behaviour:
            r = 1 / this.passenger_behaviour[x][0]
        exp_rates.append(r)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[x for x in range(sim_steps)], y=exp_rates,
                             mode='lines'))
    fig.update_layout(title="Passenger spawnrate over time",
                      xaxis_title="simulation step",
                      yaxis_title="exponential rate parameter")
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
