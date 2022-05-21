from simulation.skyscraper import Skyscraper

sky = Skyscraper()
sky.run_simulation(time=8640)
print(sky.statistics())
