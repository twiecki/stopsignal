import stopsignal
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('data_participant1.csv')

model = stopsignal.StopSignal(data)
model.find_starting_values()
model.sample(20000, burn=15000)

model.plot_posteriors()

plt.show()
