import dash_mantine_components as dmc
import pandas as pd
from components.header import MainHeader
from utils.parameter_preparation import create_behaviour_json
from components.menu import EnvironmentParameterMenu, PassengerBehaviourMenu
from components.visuals import PassengerSpawnratePlot
from dash import Dash, Output, Input
from dash_iconify import DashIconify
import plotly.graph_objects as go

from simulation.skyscraper import Skyscraper

pd.options.plotting.backend = "plotly"
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
                    children=[
                        dmc.Center(
                            dmc.Group(
                                direction="row",
                                children=[
                                    EnvironmentParameterMenu,
                                    PassengerBehaviourMenu,
                                    PassengerSpawnratePlot
                                ],
                                style={"marginTop": 20},
                            ),
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
                    ],
                    label="Run Simulation",
                    icon=[
                        DashIconify(
                            icon="el:play",
                            color=dmc.theme.DEFAULT_COLORS["green"][6],
                            width=20,
                        )
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


@app.callback(
    Output(component_id="test-output", component_property="children"),
    Input(component_id="input-simulation-steps", component_property="value"),
    Input("simulation-button", "n_clicks"),
)
def get_simulation_data(seconds_per_step, n_clicks):
    # only run if button has been pressed
    if n_clicks is not None:
        sky = Skyscraper()
        sky.run_simulation(time=8640)
        return sky.statistics()


@app.callback(
    Output(component_id="spawn-plot", component_property="figure"),
    Input(component_id="passenger-spawn-radiogroup", component_property="value"),
    Input(component_id="passenger-floor-radiogroup", component_property="value"),
    Input(component_id="input-simulation-steps", component_property="value"),
)
def update_spawn_behaviour_visual(spawn_behaviour, floor_behaviour, seconds_per_step):
    checkpoints = create_behaviour_json(seconds_per_step, spawn_behaviour, floor_behaviour)
    sim_steps = int(1440*(60/seconds_per_step))
    exp_rates = []
    r = 1 / checkpoints[0][0]
    for x in range(sim_steps):
        if x in checkpoints:
            r = 1 / checkpoints[x][0]
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
