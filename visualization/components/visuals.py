import dash_mantine_components as dmc
from dash import dcc

PassengerSpawnratePlot = dmc.Container(
    fluid=True,
    children=[
        dcc.Graph(
            id="spawn-plot"
        )
    ]
)


