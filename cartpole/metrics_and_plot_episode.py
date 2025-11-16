#!/usr/bin/env python3

import sys
import os
import h5py
import numpy as np

from matplotlib import pyplot as plt
from psipy.rl.io.batch import Episode, Batch
from psipy.rl.visualization.cartpole_plot import CartPoleTrajectoryPlot

def metrics_from_episode(episode, max_steps=None):

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

    # Further we calculate the follwoing metrics:
    # - avg and std pole angle in late phase (signed! should be zero mean, 
    #   will show any biases)
    # - avg and std costs in early as well as in late phase
    # - lower n, the steps needed to enter the tolerance range the very first
    #   time, may leave it afterwards again.


    early_threshold = 200   # end of early phase and begin of late
                            # phase after this number of steps

    costs = episode.costs if max_steps is None else episode.costs[:max_steps]
    pole_angles = episode.observations[:, 5] if max_steps is None else episode.observations[:max_steps, 5] # pole_angle is at index 5

    print(f"Evaluating {len(costs)} steps")

    deviation_bound = 10.0 / 180.0 * np.pi   # 10 degrees to radians


    # n: Find the step where the pole angle first enters the tolerance range
    n = np.where(np.abs(pole_angles) < deviation_bound)[0][0] if len(np.where(np.abs(pole_angles) < deviation_bound)[0]) > 0 else None

    # N: Find latest high angle step

    high_angle_steps = np.where(np.abs(pole_angles) >= deviation_bound)[0]  # returns all steps(-indices) outside the deviation bound. hafner and riedmiller define N with the deviation to be SMALLER (<, not <=) than the bound, pole_angle is in radians.
    latest_high_angle = high_angle_steps[-1] if len(high_angle_steps) > 0 else None
    N = latest_high_angle + 1 if latest_high_angle is not None and latest_high_angle < len(costs) else None

    # Early phase KPIs (first 200 steps)
    early_costs = costs[:early_threshold] if len(costs) > early_threshold else costs
    early_avg_cost = np.mean(early_costs)
    early_std_cost = np.std(early_costs)
    early_max_cost = np.max(early_costs)
    
    # Late phase KPIs (after 200 steps)
    late_costs = costs[early_threshold:] if len(costs) > early_threshold else []
    late_avg_cost = np.mean(late_costs) if len(late_costs) > 0 else None
    late_std_cost = np.std(late_costs) if len(late_costs) > 0 else None
    late_max_cost = np.max(late_costs) if len(late_costs) > 0 else None
    
    # Average cost per step and its standard deviation
    avg_cost_per_step = np.mean(costs) if len(costs) > 0 else None
    std_cost_per_step = np.std(costs) if len(costs) > 0 else None


    # Find latest high cost step (this checks whether or not the cart! left the # goal area)
    high_cost_steps = np.where(costs >= 0.01)[0]
    latest_high_cost = high_cost_steps[-1] if len(high_cost_steps) > 0 else None
    
    # e_T a: Find the absolute maximum angle after first 200 steps but print its signed value (positive or negative), knwoing that referenc trajectory is zero.
    e_T_200 = pole_angles[early_threshold:][np.argmax(np.abs(pole_angles[early_threshold:]))] * 180 / np.pi if len(pole_angles) > early_threshold else None

    # e_T b: Find the absolute maximum angle after N steps but print its signed value (positive or negative), knwoing that referenc trajectory is zero.
    e_T_N = pole_angles[N+20:][np.argmax(np.abs(pole_angles[N+20:]))] * 180 / np.pi if N is not None and len(pole_angles) > N+20 else None


    # Find the average angle in degrees and its standard deviation after first 200 steps. Attention: this is signed! Thus, it's not the mean of the deviation e_inf_200, but it helps to identify any biases.
    avg_angle_after_200 = np.mean(pole_angles[early_threshold:]) * 180 / np.pi if len(pole_angles) > early_threshold else None
    std_angle_after_200 = np.std(pole_angles[early_threshold:]) * 180 / np.pi if len(pole_angles) > early_threshold else None


    # e_inf a: Find the average angle in degrees and its standard deviation after 200 steps
    e_inf_200 = np.mean(abs(pole_angles[early_threshold:])) * 180 / np.pi if len(pole_angles) > early_threshold else None
    e_inf_std_200 = np.std(abs(pole_angles[early_threshold:])) * 180 / np.pi if len(pole_angles) > early_threshold else None 

    # e_inf b: Find the average angle in degrees and its standard deviation after N steps
    e_inf_N = np.mean(abs(pole_angles[N+20:])) * 180 / np.pi if N is not None and len(pole_angles) > N+20 else None
    e_inf_std_N = np.std(abs(pole_angles[N+20:])) * 180 / np.pi if N is not None and len(pole_angles) > N+20 else None    

    
    
    # Print results
    print("\nKPI Results:")

    print("\nOverall:")
    print(f"  Average cost: {avg_cost_per_step:.7f}")
    print(f"  Cost std dev: {std_cost_per_step:.7f}")

    print(f"\nEarly phase (first {early_threshold} steps):")
    print(f"  Average cost: {early_avg_cost:.7f}")
    print(f"  Cost std dev: {early_std_cost:.7f}")
    
    print(f"\nLate phase (after {early_threshold} steps):")
    if late_avg_cost is not None:
        print(f"  Average cost: {late_avg_cost:.7f}")
        print(f"  Cost std dev: {late_std_cost:.7f}")
        print(f"  Maximum cost: {late_max_cost:.7f}")
    else:
        print(f"  No data available (episode shorter than {early_threshold} steps)")

    if avg_angle_after_200 is not None:
        print(f"  Average angle: {avg_angle_after_200:.2f} degrees")
        print(f"  Angle std dev: {std_angle_after_200:.2f} degrees")
    else:
        print(f"  Average angle: Not available")
        print(f"  Angle std dev: Not available")
    

    print(f"\nLatest occurrences:")
    if latest_high_cost is not None:
        print(f"  Cost >= 0.01: Step {latest_high_cost}")
    else:
        print("  Cost >= 0.01: Never occurred")
        
    if latest_high_angle is not None:
        print(f"  |Pole angle| >= 0.1 pi: Step {latest_high_angle}")
    else:
        print("  |Pole angle| >= 0.1 pi: Never occurred")


    print(f"\nMetrics from Hafner & Riedmiller (2011):")

    print(f"  n: steps needed to enter the tolerance range for the first time.")
    print(f"  N: steps needed to enter the tolerance range (and never leave it again).")
    print(f"  e_inf: mean absolute angle deviation from 0 target angle in balancing phase.")
    print(f"  e_T: maximum absolute angle deviation from 0 target angle in balancing phase.")
    print(f"  e_inf and e_T both calculated for a T_0 = 200 and T_0 = T_N+20.\n")

    if n is not None:
        print(f"  n: {n}")
    else:
        print(f"  n: Not available")

    if N is not None:
        print(f"  N ({deviation_bound*180/np.pi} degrees): {N}")
    else:
        print(f"  N: Not available")


    if e_inf_200 is not None:
        print(f"  e_inf_200: {e_inf_200:.7f} degrees")
        print(f"  e_inf_std_200: {e_inf_std_200:.7f} degrees")
    else:
        print(f"  e_inf_200: Not available")

    if e_T_200 is not None:
        print(f"  e_T_200: {e_T_200:.7f} degrees")
    else:
        print(f"  e_T_200: Not available")

    print("\n")

    if e_inf_N is not None:
        print(f"  e_inf_N: {e_inf_N:.7f} degrees")
        print(f"  e_inf_std_N: {e_inf_std_N:.7f} degrees")
    else:
        print(f"  e_inf_N: Not available")


    if e_T_N is not None:
        print(f"  e_T_N: {e_T_N:.7f} degrees")
    else:
        print(f"  e_T_N: Not available")

    if latest_high_cost is not None and latest_high_cost > N:
        print(f"\nATTENTION: The cart left the goal area at step {latest_high_cost}!")





def plot_episode(episode, episode_num=None, max_steps=None, filename=None):
    """Load and plot a specific episode from an HDF5 file."""

#        "cart_position",    
#        "cart_velocity",
#        "pole_sine",
#        "pole_cosine",
#        "pole_velocity",
#        "pole_angle",
#        "dist_left",
#        "dist_right",
#        "direction_ACT",
    
    plot = CartPoleTrajectoryPlot(do_display=True,
                                  pole_angle_idx=5,
                                  pole_sine_idx=2,
                                  pole_cosine_idx=3,
                                  pole_velocity_idx=4,
                                  cart_position_min=0,
                                  cart_position_max=6800,
                                  cart_position_target_min=2364,
                                  cart_position_target_max=4390,
                                  max_steps=max_steps,
                                  no_title=True)

    plot.update(episode)
    plot.plot()
    plot.save(filename)
    plt.show()



def main():
    if len(sys.argv) < 2:
        print("Usage: plot_episode.py <hdf5_file>")
        sys.exit(1)
        
    filename = sys.argv[1]
    
    max_steps = None
    if len(sys.argv) > 2:
        max_steps = int(sys.argv[2])

    if not os.path.exists(filename):
        print(f"File {filename} not found")
        sys.exit(1)

    episode = Episode.from_hdf5(filename)

    print(f"Evaluating {filename}")

    metrics_from_episode(episode, max_steps=max_steps if max_steps is not None else None)
        
    plot_episode(episode, max_steps=max_steps if max_steps is not None else None, filename="episode.pdf")

if __name__ == "__main__":
    main()
