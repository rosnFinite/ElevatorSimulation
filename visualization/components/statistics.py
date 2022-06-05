import dash_mantine_components as dmc
from dash import dcc

"""
Center
    Group
        Paper
            Group
                Text
                Text
                
    Graph
    Graph
"""

TextualStats = dmc.Center(
    style={"width": "90vw"},
    children=[
        dmc.Group(
            direction="column",
            position="center",
            children=[
                dmc.Paper(
                    shadow="xs",
                    p="md",
                    withBorder=True,
                    style={},
                    children=[
                        dmc.Text("Statistics", size="lg", weight=700, align="center"),
                        dmc.Divider(variant="solid"),
                        dmc.Space(h=30),
                        dmc.Group(
                            direction="row",
                            children=[
                                dmc.Group(
                                    direction="column",
                                    grow=True,
                                    children=[
                                        dmc.Text("#Passengers created: [RUN SIMULATION FIRST]",
                                                 id="text-total-spawned"),
                                        dmc.Text("#Passengers transported: [RUN SIMULATION FIRST]",
                                                 id="text-total-transported"),
                                        dmc.Text("#Passengers abandoned: [RUN SIMULATION FIRST]",
                                                 id="text-total-abandoned"),
                                        dmc.Text("Quota: [RUN SIMULATION FIRST]",
                                                 id="text-percentage-transported")
                                        ]
                                ),
                                dmc.Group(
                                    direction="column",
                                    grow=True,
                                    children=[
                                        dmc.Text("Queue Time", weight=700),
                                        dmc.Text("Mean: [RUN SIMULATION FIRST]",
                                                 id="text-mean-queue-time",
                                                 style={"marginTop": -20}),
                                        dmc.Text("Median: [RUN SIMULATION FIRST]",
                                                 id="text-median-queue-time",
                                                 style={"marginTop": -20}),
                                        dmc.Text("Std. Deviation: [RUN SIMULATION FIRST]",
                                                 id="text-deviation-queue-time",
                                                 style={"marginTop": -20})
                                    ]
                                ),
                                dmc.Group(
                                    direction="column",
                                    grow=True,
                                    children=[
                                        dmc.Text("Elevator Time", weight=700),
                                        dmc.Text("Mean: [RUN SIMULATION FIRST]",
                                                 id="text-mean-elevator-time",
                                                 style={"marginTop": -20}),
                                        dmc.Text("Median: [RUN SIMULATION FIRST]",
                                                 id="text-median-elevator-time",
                                                 style={"marginTop": -20}),
                                        dmc.Text("Std. Deviation: [RUN SIMULATION FIRST]",
                                                 id="text-deviation-elevator-time",
                                                 style={"marginTop": -20})
                                    ]
                                )
                            ]
                        )
                    ]
                ),
                dmc.Group(
                    direction="row",
                    children=[
                        dcc.Graph(
                            id="queue-time-dist"
                        ),
                        dcc.Graph(
                            id="elevator-time-dist"
                        )
                    ]
                ),
                dmc.Group(
                    direction="row",
                    children=[
                        dcc.Graph(
                            id="queue-up-plot"
                        ),
                        dcc.Graph(
                            id="queue-down-plot"
                        )
                    ]
                )
            ]
        )
    ]
)