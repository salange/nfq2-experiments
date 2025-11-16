#!/usr/bin/env python3

import sys
import os
import h5py
import re
from psipy.rl.io.batch import Episode

def process_folder(folder_path):
    metrics = {}
    
    # Get all h5 files in the folder
    h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    
    for h5_file in h5_files:
        # Extract episode number from filename (matches -X-00.h5 pattern)
        match = re.search(r'-(\d+)-00\.h5$', h5_file)
        if not match:
            continue
            
        episode_num = int(match.group(1))
        file_path = os.path.join(folder_path, h5_file)
        
        # Load episode data
        episode = Episode.from_hdf5(file_path)
        
        # Calculate metrics
        total_cost = sum(episode.costs)
        cycles_run = len(episode.costs)
        
        # Store metrics in the same format as metrics-latest
        metrics[episode_num] = [{
            'total_cost': float(total_cost),  # Convert numpy float to Python float
            'cycles_run': cycles_run,
            'wall_time_s': 0.0  # Not available from h5 files, set to 0
        }]
    
    # Print the metrics dictionary in a format compatible with eval()
    print(metrics)

def main():
    if len(sys.argv) != 2:
        print("Usage: recalc_metrics.py <sart-eval-folder>")
        sys.exit(1)
        
    folder_path = sys.argv[1]
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a directory")
        sys.exit(1)
        
    process_folder(folder_path)

if __name__ == "__main__":
    main()