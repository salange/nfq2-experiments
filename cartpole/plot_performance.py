#! /usr/bin/env python3

import json
import os
from pprint import pprint
import sys
import numpy as np
import matplotlib.pyplot as plt


# collect the arguments given into a list of filenames
filenames = sys.argv[1:]

all_costs = []
all_episodes = []

for filename in filenames:
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

    # Get the minimum and maximum episode numbers
    min_episode = min(results.keys())
    max_episode = max(results.keys())
    
    # Create array to store ordered results (adjusted for min_episode offset)
    episode_count = max_episode - min_episode + 1
    ordered_results = [None] * episode_count
    
    # Fill array with results (adjusting indices for min_episode offset)
    for episode, result in results.items():
        ordered_results[episode - min_episode] = result
        
    # Check for missing episodes
    missing = [i + min_episode for i in range(episode_count) if ordered_results[i] is None]
    if missing:
        print(f"Warning: Missing episodes in {filename}: {missing}")
        sys.exit(1)
        
    # Replace the unordered results with ordered array (using adjusted indices)
    results = {i + min_episode: ordered_results[i] for i in range(episode_count)}

    # calculate the average step cost for each episode
    total_costs = [item[0]['total_cost'] for item in results.values()]
    cycles_run = [item[0]['cycles_run'] for item in results.values()]

    avg_step_costs = [total_costs[i] / cycles_run[i] for i in range(len(total_costs))]
    episodes = list(results.keys())  # Changed to ensure correct episode numbers
    
    # Store the data for each file
    all_costs.append(avg_step_costs)
    all_episodes.append(episodes)

# Find the shortest episode length
min_episodes = min(len(eps) for eps in all_episodes)

# Trim all data to the shortest length
trimmed_costs = [costs[:min_episodes] for costs in all_costs]

# Convert to numpy array for easier calculations
costs_array = np.array(trimmed_costs)

# Calculate mean and standard deviation across all files
mean_costs = np.mean(costs_array, axis=0)
std_costs = np.std(costs_array, axis=0)
episode_range = range(min_episodes)

# Create the plot
plt.figure(figsize=(10, 6))

# use log scale for the y-axis
plt.yscale('log')

plt.ylim(top=1e-1)
plt.ylim(bottom=1e-3)




# Plot mean with standard deviation
plt.fill_between(episode_range, mean_costs - std_costs, mean_costs + std_costs, 
                 color='blue', alpha=0.3, label='Standard Deviation')

# Plot individual runs in light grey
for i, costs in enumerate(trimmed_costs):
    plt.plot(episode_range, costs, color='grey', alpha=.5, linewidth=0.5, label='Individual runs' if i==0 else None)
#    plt.plot(episode_range, costs, color='black', alpha=0.5, linewidth=1, label='Individual runs' if i==0 else None)


plt.plot(episode_range, mean_costs, color='blue', linewidth=1, label='Mean')


plt.xlabel('Episode')
plt.ylabel('Average Cost per Step')
#plt.title('Performance in Evaluation Episodes')
plt.legend()
plt.grid(True, alpha=0.3)

# plt.show()
plt.savefig('performance.pdf', bbox_inches='tight')


