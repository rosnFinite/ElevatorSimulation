import plotly.graph_objects as go

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
                      xaxis_title="minute",
                      yaxis_title="#Passengers")
    return fig
