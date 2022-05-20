import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1(children="Elevator Simulation",),
        html.P(
            children="Analyze the behavior of avocado prices"
            " and the number of avocados sold in the US"
            " between 2015 and 2018",
        ),
        html.Div(
            children=[
                
            ]
        )
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
