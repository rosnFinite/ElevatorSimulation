import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

def plot_waiting(simulation_instance, direction):
    fig = go.Figure()
    if direction == "up":
        log = simulation_instance.get_queue_up_log
    else:
        log = simulation_instance.get_queue_down_log
    for floor in range(15):
        fig.add_trace(go.Scatter(x=[x for x in range(len(log[floor]))],
                                 y=log[floor],
                                 mode="lines",
                                 name=f'Floor {floor}'))
    fig.update_layout(title="Waiting in Queue ('UP')",
                      xaxis_title="simulation time",
                      yaxis_title="#Passengers",
                      width=800)
    return fig


def plot_distribution(value_list, title):
    mean = np.mean(value_list)
    std_plus = np.std(value_list)
    std_minus = np.std(value_list)*-1

    df = pd.DataFrame(value_list, columns=["time[s]"])

    fig = px.histogram(df, x="time[s]", marginal="box", hover_data=df.columns, title=title, width=800)

    return fig
