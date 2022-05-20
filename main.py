import datetime
from simulation.skyscraper import Skyscraper

sky = Skyscraper(random_seed=12345)
# time = 8640  // 1 sim step = 10 sec
sky.run_simulation(8640)
sky.plot_data()
avg_total = sky.get_avg_total_time()
avg_transport = sky.get_avg_transportation_time()
print('========================================')
print(f'Anzahl getätigter Anfragen {sky.num_generated_passengers}')
print(f'Anzahl erfüllter Anfragen: {sky.num_transported_passengers}')
print(f'Durchn. Zeit zum Ziel: {datetime.timedelta(seconds=avg_total)}')
print(f'Durchn. Zeit gefahren: {datetime.timedelta(seconds=avg_transport)}')
print(f'Durchn. Zeit gewartet: {datetime.timedelta(seconds=avg_total - avg_transport)}')
print('========================================')