import dash_mantine_components as dmc
from components.header import MainHeader
from components.menu import SimulationParameterMenu
from dash import Dash, Output, Input

from simulation.skyscraper import Skyscraper

app = Dash(__name__)

app.layout = dmc.Container(
    children=[
        MainHeader,
        dmc.Group(
            children=[SimulationParameterMenu],
            style={"marginTop": 20},
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


if __name__ == "__main__":
    app.run_server()
