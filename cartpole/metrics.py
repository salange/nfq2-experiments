#!/usr/bin/env python3

import sys
import os
import h5py
import numpy as np
import hashlib
import glob

from psipy.rl.io.batch import Episode


def calculate_metrics(episode, max_steps=None):
    """Calculate metrics for a single episode."""

    # We calculate some metrics including those from Hafner & Riedmiller
    # "Reinforcement Learning in Feedback Control" (2011) which are:
    #
    # N - number of steps needed to reach the tolearance range(goal state) and #     never leave it again
    # e_inf - mean absolute angle deviation after a) a fixed early phase of 200 #         steps and b) after N (from above)
    # e_T - here as the maximum absolute angle deviation after a) a fixed early 
    #       phase of 200 steps and b) after N+20 (from above) steos. Reasoning #       is:
    # 
    #       we don't know the optimal reference trajectory during swing up
    #       but only for the balancing phase, where it is zero (for the 
    #       pole angle). In paper we'd use e_T_200 as the default, but use
    #       e_T_N where N is larger than 180 (thus, equilibrium was not reached #       yet in 200-20 steps). 20 steps are the period we let the controller
    #       take to move the pole from tolerance bound to zero angle goal state.
    #
    # We calculate these for the pole angle only.

    costs = episode.costs if max_steps is None else episode.costs[:max_steps]
    pole_angles = episode.observations[:, 5] if max_steps is None else episode.observations[:max_steps, 5]
    
    deviation_bound = 10.0 / 180.0 * np.pi
    early_threshold = 200

    # Calculate n (first entry into tolerance range)
    tolerance_indices = np.where(np.abs(pole_angles) < deviation_bound)[0]
    n = tolerance_indices[0] if len(tolerance_indices) > 0 else None

    # Calculate N (final entry into tolerance range)
    high_angle_steps = np.where(np.abs(pole_angles) >= deviation_bound)[0]
    N = high_angle_steps[-1] + 1 if len(high_angle_steps) > 0 else None

    # Calculate e_inf and e_T after 200 steps
    e_inf_200 = np.mean(abs(pole_angles[early_threshold:])) * 180 / np.pi if len(pole_angles) > early_threshold else None
    e_T_200 = np.abs(pole_angles[early_threshold:][np.argmax(np.abs(pole_angles[early_threshold:]))]) * 180 / np.pi if len(pole_angles) > early_threshold else None

    # Calculate e_inf and e_T after N+20 steps
    e_inf_N = np.mean(abs(pole_angles[N+20:])) * 180 / np.pi if N is not None and len(pole_angles) > N+20 else None
    e_T_N = np.abs(pole_angles[N+20:][np.argmax(np.abs(pole_angles[N+20:]))]) * 180 / np.pi if N is not None and len(pole_angles) > N+20 else None

    # Calculate average cost
    avg_cost = np.mean(costs)

    return {
        'n': n,
        'N': N,
        'e_inf_200': e_inf_200,
        'e_T_200': e_T_200,
        'e_inf_N': e_inf_N,
        'e_T_N': e_T_N,
        'avg_cost': avg_cost,
        'cycles_run': len(costs),
        'total_cost': sum(costs)
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: metrics.py <folder_path> [max_steps]")
        sys.exit(1)

    folder_path = sys.argv[1]
    max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else None

    # Get all h5 files in the folder
    h5_files = glob.glob(os.path.join(folder_path, '*-*.h5'))
    results = []

    for file_path in h5_files:
        # Extract episode number from filename
        episode_num = int(file_path.split('-')[-2])
        episode = Episode.from_hdf5(file_path)
        metrics = calculate_metrics(episode, max_steps)
        results.append((episode_num, metrics))

    # Sort results by episode number
    results.sort(key=lambda x: x[0])

    # Calculate averages and standard deviations
    metrics_array = {
        'n': [r[1]['n'] for r in results if r[1]['n'] is not None],
        'N': [r[1]['N'] for r in results if r[1]['N'] is not None],
        'e_inf_200': [r[1]['e_inf_200'] for r in results if r[1]['e_inf_200'] is not None],
        'e_T_200': [r[1]['e_T_200'] for r in results if r[1]['e_T_200'] is not None],
        'e_inf_N': [r[1]['e_inf_N'] for r in results if r[1]['e_inf_N'] is not None],
        'e_T_N': [r[1]['e_T_N'] for r in results if r[1]['e_T_N'] is not None],
        'avg_cost': [r[1]['avg_cost'] for r in results if r[1]['avg_cost'] is not None]
    }

    # Calculate and print statistics
    print("\n# Metrics Statistics:")
    for metric, values in metrics_array.items():
        if values:
            print(f"# {metric}: {np.mean(values):.4f} +- {np.std(values):.4f}")

    # Find and print the 10 episodes with lowest N where the episode length is at least 200, lowest to highest, format: episode_num: N
    lowest_N_episodes = sorted([r for r in results if r[1]['N'] is not None and r[1]['cycles_run'] >= 200], key=lambda x: x[1]['N'])[:10]
    print("\n# 10 episodes with lowest N:")
    for episode_num, metrics in lowest_N_episodes:
        print(f"# {episode_num}: {metrics['N']}")

    # Find and print the 10 episodes with lowest e_inf_200, lowest to highest, format: episode_num: e_inf_200
    lowest_e_inf_200_episodes = sorted([r for r in results if r[1]['e_inf_200'] is not None], key=lambda x: x[1]['e_inf_200'])[:10]
    print("\n# 10 episodes with lowest e_inf_200:")
    for episode_num, metrics in lowest_e_inf_200_episodes:
        print(f"# {episode_num}: {metrics['e_inf_200']}")

    # Find and print the 10 episodes with lowest e_T_200 (absolute value), lowest to highest, format: episode_num: e_T_200
    lowest_e_T_200_episodes = sorted([r for r in results if r[1]['e_T_200'] is not None], key=lambda x: abs(x[1]['e_T_200']))[:10]
    print("\n# 10 episodes with lowest e_T_200:")
    for episode_num, metrics in lowest_e_T_200_episodes:
        print(f"# {episode_num}: {metrics['e_T_200']}")
        
    # Create output dictionary in the format shown above
    output = {}
    for episode_num, metrics in results:
        output[episode_num] = [{
            'cycles_run': metrics['cycles_run'],
            'total_cost': metrics['total_cost'],
            'n': metrics['n'],
            'N': metrics['N'],
            'e_inf_200': metrics['e_inf_200'],
            'e_T_200': metrics['e_T_200'],
            'e_inf_N': metrics['e_inf_N'],
            'e_T_N': metrics['e_T_N'],
            'avg_cost': metrics['avg_cost'],
            'wall_time_s': 0.0  # Not tracking wall time
        }]

    # Save the output dictionary to a file metrics in the folder
    with open(os.path.join(folder_path, 'metrics'), 'w') as f:
        f.write(str(output))

if __name__ == "__main__":
    main()


