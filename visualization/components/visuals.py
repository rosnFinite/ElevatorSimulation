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

TimeLine = dmc.Container(
    size="xl",
    children=[
        dmc.Slider(
            id="drag-slider",
            size="xl",
            min=0,
            value=0,
            step=1,
            max=8640,
            style={"width": "100%"}
        ),
        dmc.Text(id="drag-value"),
        dmc.Space(h=30),
        dcc.Graph(id="visual-plot")
    ]
)
