"""
Dash components for rendering the header
"""
import dash_mantine_components as dmc
from dash import html, dcc
from dash_iconify import DashIconify


def create_home_link(label):
    return dmc.Text(
        label,
        size="xl",
        color="gray",
    )


MainHeader = dmc.Header(
    height=80,
    p="md",
    children=[
        dmc.Container(
            fluid=True,
            children=dmc.Group(
                position="apart",
                align="flex-start",
                children=[
                    dmc.Center(
                        dcc.Link(
                            [
                                dmc.Group(
                                    children=[
                                        dmc.Image(
                                          src="../assets/SimLogo.png", alt="Logo", width=40
                                        ),
                                        dmc.MediaQuery(
                                            create_home_link("Elevator Simulation"),
                                            smallerThan="sm",
                                            styles={"display": "none"},
                                        ),
                                    ]
                                )
                            ],
                            href="/",
                            style={"textDecoration": "none"},
                        ),
                    ),
                    dmc.Group(
                        position="right",
                        align="center",
                        spacing="xl",
                        children=[
                            html.A(
                                dmc.Tooltip(
                                    dmc.ThemeIcon(
                                        DashIconify(
                                            icon="radix-icons:github-logo",
                                            width=30,
                                        ),
                                        radius=30,
                                        size=36,
                                        variant="outline",
                                        color="gray",
                                    ),
                                    label="Source Code",
                                    position="bottom",
                                ),
                                href="https://github.com/rosnFinite/ElevatorSimulation",
                            ),
                        ]
                    )
                ]
            )
        )
    ]
)
