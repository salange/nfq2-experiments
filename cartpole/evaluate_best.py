#! /usr/bin/env python3

import json
import os
from pprint import pprint
import sys
import numpy as np
import matplotlib.pyplot as plt


folder = sys.argv[1]
filename = os.path.join(folder, 'metrics-latest')

# load the hash that was written by pprint from the file
with open(filename, 'r') as f:
    data = f.read()

# convert the hash to a dictionary  
results = eval(data)

# resutls has the structure:
# {0: [{'cycles_run': 104,
#       'total_cost': 2.059904454854694,
#       'wall_time_s': 0.6996}],
# ...
# }


costs_per_step = [item[0]['total_cost'] / item[0]['cycles_run'] for item in results.values()]


# calculate and print the minimum cost per step and its episode
min_cost_episode = np.argmin(costs_per_step)
min_cost = costs_per_step[min_cost_episode]

print(f"Minimum cost per step: {min_cost} in episode {min_cost_episode}\n")

# Run metrics and plotting on the best episode
os.system(f"python3 metrics_and_plot_episode.py {folder}/sart-eval/Cartpole*-{min_cost_episode}-00.h5")




