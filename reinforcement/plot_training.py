import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def read_json_data_as_df(filepath):
    with open(filepath, "r") as infile:
        data = json.load(infile)
    return pd.DataFrame.from_dict(data)


def plot_ep_rewards(dataframe):
    fig = px.line(dataframe, x=dataframe.index.array, y=dataframe["ep_rewards"], title="Rewards per Episode")
    fig.show()


def plot_policy_loss(dataframe):
    fig = px.line(dataframe, x=dataframe.index.array, y=dataframe["policy_loss"], title="Policy Loss")
    fig.show()


def plot_value_loss(dataframe):
    fig = px.line(dataframe, x=dataframe.index.array, y=dataframe["value_loss"], title="Value Loss")
    fig.show()


def plot_total_loss(dataframe):
    fig = px.line(dataframe, x=dataframe.index.array, y=dataframe["total_loss"], title="Total Loss")
    fig.show()


def show_combined_plot(dataframe):
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Rewards", "Policy Loss", "Value Loss", "Total Loss"])

    x = dataframe.index.array
    fig.add_trace(go.Scatter(x=x, y=dataframe["ep_rewards"]), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=dataframe["policy_loss"]), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=dataframe["value_loss"]), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=dataframe["total_loss"]), row=2, col=2)

    fig.show()


df = read_json_data_as_df("models/one_elevator_test/metrics/ep_1000.json")
show_combined_plot(df)








