import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots


def plot_waiting(simulation_instance, direction):
    fig = go.Figure()
    if direction == "up":
        log = simulation_instance.df_log.loc[:, "up_0":"up_14"]
    else:
        log = simulation_instance.df_log.loc[:, "down_0":"down_14"]
    for index, column in enumerate(log.columns):
        fig.add_trace(go.Scatter(x=log.index,
                                 y=log[column],
                                 mode="lines",
                                 name=f'Floor {index}'))
    fig.update_layout(title=f'Passengers Waiting in Queue ({direction})',
                      xaxis_title="simulation step",
                      yaxis_title="#Passengers",
                      width=800)
    return fig


def plot_distribution(value_list, title):
    df = pd.DataFrame(value_list, columns=["time[s]"])
    fig = px.histogram(df, x="time[s]", marginal="box", hover_data=df.columns, title=title, width=800)
    return fig


def plot_barchart(simulation_instance, time):
    floor_name = [str(x) for x in range(15)]
    # get data for specified time and direction
    fig = make_subplots(rows=1, cols=3, column_width=[0.2, 0.6, 0.2],
                        subplot_titles=("Q-Up", "Elevator", "Q-Down"),
                        horizontal_spacing=0,
                        shared_yaxes=True)
    log_up = simulation_instance.df_log.loc[time, "up_0":"up_14"].to_frame()
    log_up.columns = ["passengers"]
    log_up.index = floor_name
    log_down = simulation_instance.df_log.loc[time, "down_0":"down_14"].to_frame()
    log_down.columns = ["passengers"]
    log_down.index = floor_name
    elevator_pos = simulation_instance.df_log.loc[time, ["e0_pos", "e1_pos", "e2_pos"]].to_list()
    elevator_util = simulation_instance.df_log.loc[time, ["e0_util", "e1_util", "e2_util"]].to_list()
    fig.add_trace(
        go.Bar(
            x=log_up["passengers"],
            y=log_up.index,
            orientation="h",
            showlegend=False,
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1, 2],
            y=elevator_pos,
            text=elevator_util,
            textposition="middle center",
            mode="markers+text",
            marker=dict(
                color="LightSkyBlue",
                size=20
            ),
            yaxis="y2",
            showlegend=False
        ),
        row=1,
        col=2
    )
    fig.add_trace(
        go.Bar(
            x=log_down["passengers"],
            y=log_down.index,
            orientation="h",
            showlegend=False
        ),
        row=1,
        col=3
    )
    fig.update_yaxes(title_text="Floor", row=1, col=1)
    fig.update_layout(yaxis2=dict(range=[-0.5, 14.5], tickmode="linear", tick0=0, dtick=1))
    return fig

