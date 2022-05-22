import dash_mantine_components as dmc
from dash import dcc

TextualStats = dmc.Group(
    direction="row",
    children=[
        dmc.Paper(
            shadow="xs",
            p="md",
            withBorder=True,
            style={},
            children=[
                dmc.Group(
                    direction="column",
                    grow=True,
                    children=[
                        dmc.Text("Statistics", size="lg", align="center"),
                        dmc.Divider(variant="solid", style={"marginTop": -20}),
                        dmc.Text("#Passengers created: [RUN SIMULATION FIRST]",
                                 id="text-total-spawned",
                                 style={"marginTop": -20}),
                        dmc.Text("#Passengers transported: [RUN SIMULATION FIRST]",
                                 id="text-total-transported",
                                 style={"marginTop": -20}),
                        dmc.Text("#Passengers abandoned: [RUN SIMULATION FIRST]",
                                 id="text-total-abandoned",
                                 style={"marginTop": -20}),
                        dmc.Text("Quota: [RUN SIMULATION FIRST]",
                                 id="text-percentage-transported",
                                 style={"marginTop": -20}),
                        dmc.Text("Queue Time", weight=700),
                        dmc.Text("Mean: [RUN SIMULATION FIRST]",
                                 id="text-mean-queue-time",
                                 style={"marginTop": -20}),
                        dmc.Text("Median: [RUN SIMULATION FIRST]",
                                 id="text-median-queue-time",
                                 style={"marginTop": -20}),
                        dmc.Text("Std. Deviation: [RUN SIMULATION FIRST]",
                                 id="text-deviation-queue-time",
                                 style={"marginTop": -20}),
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
        ),
        dcc.Graph(
            id="queue-up-plot"
        ),
        dcc.Graph(
            id="queue-down-plot"
        )
    ]
)