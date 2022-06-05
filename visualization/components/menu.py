import dash_mantine_components as dmc

EnvironmentParameterMenu = dmc.Paper(
    shadow="xs",
    p="md",
    withBorder=True,
    style={"height": 400, "width": "100%"},
    children=dmc.Group(
        grow=True,
        direction="column",
        children=[
            dmc.Text("Environment Parameter", size="lg", weight=700, align="center"),
            dmc.Divider(variant="solid", style={"marginTop": -20}),
            dmc.Space(h=30),
            dmc.Center(
                dmc.Group(
                    direction="column",
                    style={"marginTop": -20},
                    children=[
                        dmc.NumberInput(
                            id="input-simulation-steps",
                            label="Seconds per Simulation Step",
                            value=10,
                            min=5,
                            max=60,
                            step=5,
                            style={"width": 250}
                        ),
                        dmc.Space(h=10),
                        dmc.NumberInput(
                            id="input-random-seed",
                            description="Seed to reproduce results [default: None]",
                            label="Random Seed",
                            value=None,
                            style={"width": 250}
                        )
                    ]
                )
            )
        ]
    )
)

PassengerBehaviourMenu = dmc.Paper(
    shadow="xs",
    p="md",
    withBorder=True,
    style={"height": 400, "width": "100%"},
    children=dmc.Group(
        grow=True,
        direction="column",
        children=[
            dmc.Text("Passenger Behaviour", size="lg", weight=700, align="center"),
            dmc.Divider(variant="solid", style={"marginTop": -20}),
            dmc.Space(h=10),
            dmc.Center(
                dmc.Group(
                    direction="column",
                    children=[
                        dmc.RadioGroup(
                            id="passenger-spawn-radiogroup",
                            data=[
                                {"value": "realism", "label": "Realism"},
                                {"value": "random", "label": "Random"},
                                {"value": "static", "label": "Static"}
                            ],
                            value="realism",
                            label="Spawn behaviour",
                            description="Time between Passenger Arrival [default: Realism]"
                        ),
                        dmc.Space(h=20),
                        dmc.RadioGroup(
                            id="passenger-floor-radiogroup",
                            data=[
                                {"value": "realism", "label": "Realism"},
                                {"value": "random", "label": "Random"}
                            ],
                            value="realism",
                            label="Floor switch behaviour",
                            description="Decides start floor and destination floor [default: Realism]"
                        )
                    ]
                )
            )
        ]
    )
)


