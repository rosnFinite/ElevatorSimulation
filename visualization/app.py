import sys
import dash_mantine_components as dmc
import datetime
import time
from dash import Dash, Output, Input, State
from dash_iconify import DashIconify
import plotly.graph_objects as go

# intern functions
import utils.plotting as plotting
from utils.parameter_preparation import create_passenger_behaviour
from components.header import MainHeader
from components.menu import EnvironmentParameterMenu, PassengerBehaviourMenu
from components.visuals import PassengerSpawnratePlot, TimeLine
from components.statistics import TextualStats
from simulation.skyscraper import Skyscraper

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]
this.skyscraper = None
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
                                dmc.Col(dmc.Center([EnvironmentParameterMenu]), span=6),
                                dmc.Col(dmc.Center([PassengerBehaviourMenu]), span=6),
                                dmc.Col(PassengerSpawnratePlot, span=12)
                            ],
                            style={"marginTop": 20},
                            gutter="xs"
                        ),
                        dmc.Container(
                            children=[
                                dmc.Button(
                                    "Start Simulation",
                                    id="simulation-button",
                                    leftIcon=[DashIconify(icon="el:play")],
                                    variant="gradient",
                                    fullWidth=True,
                                    gradient={"from": "teal", "to": "lime", "deg": 105},
                                    size="md",
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
                        TimeLine
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
    return f'Total simulation steps: {int(1440 * (60 / seconds_per_step))}'


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
        Output(component_id="queue-down-plot", component_property="figure"),
        Output(component_id="queue-time-dist", component_property="figure"),
        Output(component_id="elevator-time-dist", component_property="figure")
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
        this.skyscraper = Skyscraper(random_seed, passenger_behaviour=this.passenger_behaviour)
    else:
        this.skyscraper = Skyscraper(passenger_behaviour=this.passenger_behaviour)
    this.skyscraper.run_simulation(time=int(1440 * (60 / seconds_per_step)))
    total_spawned = f'#Passengers created: {this.skyscraper.num_generated_passengers}'
    total_transported = f'#Passengers transported: {this.skyscraper.num_transported_passengers}'
    total_abandoned = f'#Passengers abandoned: {this.skyscraper.num_generated_passengers - this.skyscraper.num_transported_passengers}'
    percentage_transported = f'Quota: {this.skyscraper.num_transported_passengers / this.skyscraper.num_generated_passengers * 100:.2f}%'
    mean_queue_time = f'Mean: {datetime.timedelta(seconds=this.skyscraper.mean_queue_time)}'
    median_queue_time = f'Median: {datetime.timedelta(seconds=this.skyscraper.median_queue_time)}'
    std_queue_time = f'Std. Deviation: {datetime.timedelta(seconds=this.skyscraper.std_queue_time)}'
    mean_elevator_time = f'Mean: {datetime.timedelta(seconds=this.skyscraper.mean_travel_time)}'
    median_elevator_time = f'Median: {datetime.timedelta(seconds=this.skyscraper.mean_travel_time)}'
    std_elevator_time = f'Std. Deviation: {datetime.timedelta(seconds=this.skyscraper.std_travel_time)}'
    fig_up = plotting.plot_waiting(this.skyscraper, "up")
    fig_down = plotting.plot_waiting(this.skyscraper, "down")
    dist_q = plotting.plot_distribution(this.skyscraper.queue_time_log, title="Queue time distribution")
    dist_el = plotting.plot_distribution(this.skyscraper.travel_time_log, title="Travel time distribution")

    return total_spawned, total_transported, total_abandoned, percentage_transported, mean_queue_time, \
           median_queue_time, std_queue_time, mean_elevator_time, median_elevator_time, std_elevator_time, fig_up, \
           fig_down, dist_q, dist_el


# -----------------------------VISUALIZE TIMELINE-----------------------------


@app.callback(
    Output(component_id="drag-value", component_property="children"),
    Output(component_id="drag-slider", component_property="max"),
    State(component_id="input-simulation-steps", component_property="value"),
    Input(component_id="drag-slider", component_property="value")
)
def convert_to_datetime(seconds_per_step, slider_value):
    slider_max = int(1440 * (60 / seconds_per_step)) - 1
    if this.skyscraper is not None:
        slider_max = len(this.skyscraper.df_log) - 1
    displayed_time = time.strftime('%H:%M:%S', time.gmtime(slider_value * seconds_per_step))
    return f'Time of day:  {displayed_time}', slider_max


@app.callback(
    Output(component_id="visual-plot", component_property="figure"),
    Input(component_id="drag-slider", component_property="value"),
    prevent_initial_call=True
)
def update_queue_barchart(slider_value):
    return plotting.plot_barchart(this.skyscraper, slider_value)


# -----------------------------UPDATE PASSENGER SPAWN RATE PLOT-----------------------------

@app.callback(
    Output(component_id="spawn-plot", component_property="figure"),
    Input(component_id="passenger-spawn-radiogroup", component_property="value"),
    Input(component_id="passenger-floor-radiogroup", component_property="value"),
    Input(component_id="input-simulation-steps", component_property="value"),
)
def update_spawn_behaviour_visual(spawn_behaviour, floor_behaviour, seconds_per_step):
    this.passenger_behaviour = create_passenger_behaviour(seconds_per_step, spawn_behaviour, floor_behaviour)
    sim_steps = int(1440 * (60 / seconds_per_step))
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
