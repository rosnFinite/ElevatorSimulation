import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


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
