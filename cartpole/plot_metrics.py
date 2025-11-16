#!/usr/bin/env python3

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(filename):
    # Load the metrics file
    with open(filename, 'r') as f:
        data = eval(f.read())
    
    # Get the maximum episode number
    max_episode = max(data.keys())
    
    # Create arrays to store ordered results
    ordered_results = [None] * (max_episode + 1)
    
    # Fill array with results
    for episode, result in data.items():
        ordered_results[episode] = result[0]
    
    # Extract metrics
    episodes = list(range(len(ordered_results)))
    n_values = [r['n'] if r['n'] is not None else np.nan for r in ordered_results]
    N_values = [r['N'] if r['N'] is not None else np.nan for r in ordered_results]
    e_T_200_values = [r['e_T_200'] if r['e_T_200'] is not None else np.nan for r in ordered_results]
    e_inf_200_values = [r['e_inf_200'] if r['e_inf_200'] is not None else np.nan for r in ordered_results]

    # Create the plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Plot n and N on the first y-axis
    ln1 = ax1.plot(episodes, n_values, 'b-', label='n (first entry)', alpha=0.7)
    ln2 = ax1.plot(episodes, N_values, 'g-', label='N (final entry)', alpha=0.7)
    
    # Plot e_T and e_inf on the second y-axis
    ln3 = ax2.plot(episodes, e_T_200_values, 'r-', label='e_T_200', alpha=0.7)
    ln4 = ax2.plot(episodes, e_inf_200_values, 'y-', label='e_inf_200', alpha=0.7)

    # Set labels and title
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Steps')
    ax2.set_ylabel('Angle (degrees)')

    # Add grid
    ax1.grid(True, alpha=0.3)

    # Combine legends from both axes
    lns = ln1 + ln2 + ln3 + ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')

    # Adjust layout to prevent label clipping
    plt.tight_layout()

    # Save the plot
    plt.savefig('metrics.pdf', bbox_inches='tight')
    plt.close()

def main():
    if len(sys.argv) != 2:
        print("Usage: plot_metrics.py <metrics_file>")
        sys.exit(1)

    metrics_file = sys.argv[1]
    if not os.path.exists(metrics_file):
        print(f"File {metrics_file} not found")
        sys.exit(1)

    plot_metrics(metrics_file)

if __name__ == "__main__":
    main()
