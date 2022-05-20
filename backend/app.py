from flask import Flask
from simulation.skyscraper import Skyscraper

app = Flask(__name__)


@app.route("/")
def get_simulation_data():
    sky = Skyscraper()
    sky.run_simulation(time=8640)
    stats = sky.statistics()
    return f'<p> {stats} </p>'


def start_backend():
    app.run()
