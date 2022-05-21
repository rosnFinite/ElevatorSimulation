import dash_mantine_components as dmc
from dash_iconify import DashIconify

SimulationParameterMenu = dmc.Paper(
    shadow="xs",
    p="md",
    withBorder=True,
    children=dmc.LoadingOverlay(
        dmc.Group(
            grow=True,
            direction="column",
            children=[
                dmc.Text("Run Simulation", size="lg", align="center"),
                dmc.Divider(variant="solid", style={"marginTop": -20}),
                dmc.Group(
                    style={"marginTop": -20},
                    children=[
                        dmc.NumberInput(
                            id="input-simulation-steps",
                            label="Seconds per Simulation Step",
                            value=10,
                            min=5,
                            max=60,
                            step=5
                        ),
                    ],
                ),
                dmc.Button(
                    "Start Simulation",
                    id="simulation-button",
                    fullWidth=True,
                    leftIcon=[DashIconify(icon="el:play")],
                    variant="gradient",
                    gradient={"from": "teal", "to": "lime", "deg": 105},
                ),
                dmc.Text(id="test-output", children="halle")
            ]
        )
    )
)
