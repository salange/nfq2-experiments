import sys
# from turtle import position
import numpy as np
from typing import Callable, List, Type
from pprint import pprint

from psipy.rl.core.plant import State
from psipy.rl.core.experimentation import Environment
from psipy.rl.visualization.cartpole_plot import CartPoleTrajectoryPlot

from psipy.rl.plants.real.pact_cartpole.cartpole import (
    SwingupContinuousDiscreteAction,
    SwingupContinuousExtendedDiscreteAction,
    SwingupPlant,
    GeneralizedCartpolePlant,
    SwingupState,
    plot_swingup_state_history
)



def make_sparse_cost_func(position_idx: int=0,
                          cosine_idx: int=3,
                          step_cost: float=0.01,
                          use_cosine: bool=True,
                          use_upright_margin: bool=False,
                          upright_margin: float=0.3,
                          xminus: bool=True) -> Callable[[np.ndarray], np.ndarray]:
    # Define a custom cost function to change the inbuilt costs
    def sparse_costfunc(states: np.ndarray) -> np.ndarray:
        center = (SwingupPlant.LEFT_SIDE + SwingupPlant.RIGHT_SIDE) / 2.0
        margin = abs(SwingupPlant.RIGHT_SIDE - SwingupPlant.LEFT_SIDE) / 2.0 * 0.3  # 30% of distance from center to hard endstop

        position = states[:, position_idx] 
        cosine = states[:, cosine_idx]       

        if isinstance(cosine, np.ndarray):
            costs = np.zeros(cosine.shape)
        else:
            costs = 0.0

        if use_cosine:
            costs = (1.0-(cosine+1.0)/2.0) * step_cost  # shaping of costs in goal area to reward low pole angle deviations from upright position

        if use_upright_margin:
            costs[1.0-(cosine+1.0)/2.0 > upright_margin] = step_cost    

        costs[abs(position - center) >= margin] = step_cost 

        if xminus:  # non-terminal bad area
            costs[position + SwingupPlant.LEFT_SIDE <= SwingupPlant.LEFT_SIDE + SwingupPlant.TERMINAL_LEFT_OFFSET * 2] = step_cost * 5
            costs[position + SwingupPlant.LEFT_SIDE >= SwingupPlant.RIGHT_SIDE - SwingupPlant.TERMINAL_RIGHT_OFFSET * 2] = step_cost * 5

        # terminal bad area
        costs[position + SwingupPlant.LEFT_SIDE <= SwingupPlant.LEFT_SIDE + SwingupPlant.TERMINAL_LEFT_OFFSET] = 1.0
        costs[position + SwingupPlant.LEFT_SIDE >= SwingupPlant.RIGHT_SIDE - SwingupPlant.TERMINAL_RIGHT_OFFSET] = 1.0

        return costs

    return sparse_costfunc



def make_sway_killer_cost_func(position_idx: int=0,
                               cosine_idx: int=3,
                               step_cost: float=0.01,
                               use_cosine: bool=False,
                               down_margin: float=0.005) -> Callable[[np.ndarray], np.ndarray]: # we use 0.001 when learning from scratch
    def sparse_costfunc(states: np.ndarray) -> np.ndarray:
        center = (SwingupPlant.LEFT_SIDE + SwingupPlant.RIGHT_SIDE) / 2.0
        margin = abs(SwingupPlant.RIGHT_SIDE - SwingupPlant.LEFT_SIDE) / 2.0 * 0.3  # 25% of distance from center to hard endstop

        position = states[:, position_idx]
        cosine = states[:, cosine_idx]

        if use_cosine:
            costs = (cosine+1.0)/2.0 * step_cost  # can only get lower costs in center of x axis
        else:
            costs = ((cosine+1.0)/2.0 > down_margin) * step_cost

        costs[abs(position - center) > margin] = 0.01 
        costs[position + SwingupPlant.LEFT_SIDE <= SwingupPlant.LEFT_SIDE + SwingupPlant.TERMINAL_LEFT_OFFSET * 2] = 0.1
        costs[position + SwingupPlant.LEFT_SIDE >= SwingupPlant.RIGHT_SIDE - SwingupPlant.TERMINAL_RIGHT_OFFSET * 2] = 0.1
        costs[position + SwingupPlant.LEFT_SIDE <= SwingupPlant.LEFT_SIDE + SwingupPlant.TERMINAL_LEFT_OFFSET] = 1.0
        costs[position + SwingupPlant.LEFT_SIDE >= SwingupPlant.RIGHT_SIDE - SwingupPlant.TERMINAL_RIGHT_OFFSET] = 1.0


        return costs

    return sparse_costfunc



def make_antisway_approach_cost_func(position_idx: int=0,
                                     cosine_idx: int=3,
                                     left_distance_idx: int=4,
                                     right_distance_idx: int=5,
                                     direction_idx: int=6,
                                     step_cost: float=0.01,
                                     use_cosine: bool=False,
                                     down_margin: float=0.001,
                                     position_margin: float=50,
                                     use_distance: bool=False,
                                     punish_direction: bool=True) -> Callable[[np.ndarray], np.ndarray]:
    
    def antisway_costfunc(states: np.ndarray) -> np.ndarray:

        position = states[:, position_idx] # we assume its zero centered
        cosine = states[:, cosine_idx]
        dist_left = states[:, left_distance_idx]
        dist_right = states[:, right_distance_idx]
        direction = states[:, direction_idx]

        dir_punishment = (abs(direction) > 0.001) * step_cost * 0.1

        if use_cosine:
            costs = (cosine+1.0)/2.0 * step_cost # cosine based cost
        else:
            costs = ((cosine+1.0)/2.0 > down_margin) * step_cost   # zero, when "down enough", step costs otherwise

        if punish_direction:
            dir_punishment = (abs(direction) > 0.001) * step_cost * 0.1
            if not use_cosine:
                dir_punishment = dir_punishment * ((cosine+1.0)/2.0 <= down_margin) # can only occur, if in goal area
            costs = costs + dir_punishment
    
       
        costs[abs(position) > position_margin] = 0.01 # zero, when "far enough"

        if use_distance:
            costs = costs + abs(position) / 5000.0 * 0.01  # roughly up to one full step on top
    
        costs[dist_left >= SwingupPlant.TERMINAL_LEFT_OFFSET] = 0.1
        costs[dist_right <= SwingupPlant.TERMINAL_RIGHT_OFFSET] = 0.1
        costs[dist_left > 0] = 1.0
        costs[dist_right < 0] = 1.0

        

        # potential additions:
        #   - penalize distance to center
        #   - constraints on maximum angle (wear and tear, crane specs)
        #   - penalize overshooting the target position
        #   - penalize any corrections after "arriving" by increaseing step costs (we want perfect arrival)

        return costs

    return antisway_costfunc





def setup_pact_cartpole(environment: Environment,
                        fast: bool=False,
                        use_extended_action_space: bool=False,
                        sway_killer: bool=False):

    if use_extended_action_space:
        environment.action_type = SwingupContinuousExtendedDiscreteAction
    else:
        environment.action_type = SwingupContinuousDiscreteAction


    environment.state_type  = SwingupState

    environment.state_channels = (
        "cart_position",
        "cart_velocity",
        "pole_sine",
        "pole_cosine",
        "pole_velocity",
        # "pole_angle",
        "direction_ACT",
    )

    environment.controller_action_channels = ["direction"]
    environment.batch_action_channels = ["direction_index"]
    environment.max_episode_length = 400
    
    if sway_killer:
        environment.cost_function = make_sway_killer_cost_func(
            position_idx=environment.state_channels.index("cart_position"),
            cosine_idx=environment.state_channels.index("pole_cosine"))
    else:
        environment.cost_function = make_sparse_cost_func(
            position_idx=environment.state_channels.index("cart_position"),
            cosine_idx=environment.state_channels.index("pole_cosine"))

    print(">>> ATTENTION: chosen cost function: ", environment.cost_function)

    if fast:
        SwingupPlant.ACTION_DELAY = 0.025
        environment.lookback = 13
    else:
        environment.lookback = 6

    environment.plant = SwingupPlant(
        hostname="127.0.0.1",
        hilscher_port="5555",
        sway_start=False,
        cost_function=SwingupPlant.cost_func_wrapper(
            environment.cost_function,
            environment.state_channels))

    environment.plot_history = plot_swingup_state_history
    environment.trajectory_plot_exploration = CartPoleTrajectoryPlot(
        cart_position_idx=0,
        cart_velocity_idx=1,
        pole_sine_idx=2,
        pole_cosine_idx=3,
        pole_velocity_idx=4)
    environment.trajectory_plot_evaluation = CartPoleTrajectoryPlot(
        cart_position_idx=0,
        cart_velocity_idx=1,
        pole_sine_idx=2,
        pole_cosine_idx=3,
        pole_velocity_idx=4)



    return environment


def setup_pact_antisway(environment: Environment,
                        fast: bool=False,
                        use_extended_action_space: bool=True):

    if use_extended_action_space:
        environment.action_type = SwingupContinuousExtendedDiscreteAction
    else:
        environment.action_type = SwingupContinuousDiscreteAction

    environment.state_type  = SwingupState

    environment.state_channels = (
        "cart_position",
        "cart_velocity",
        "pole_sine",
        "pole_cosine",
        "pole_velocity",
        "dist_left",
        "dist_right",
        "direction_ACT",
    )

    environment.controller_action_channels = ["direction"]
    environment.batch_action_channels = ["direction_index"]
    environment.max_episode_length = 80 # 200 # 400
    
    environment.cost_function = make_antisway_approach_cost_func(
        position_idx=environment.state_channels.index("cart_position"),
        cosine_idx=environment.state_channels.index("pole_cosine"),
        left_distance_idx=environment.state_channels.index("dist_left"),
        right_distance_idx=environment.state_channels.index("dist_right"),
        direction_idx=environment.state_channels.index("direction_ACT"),
        position_margin=30,
        down_margin=0.002,
        use_distance=True,
        use_cosine=False)

    print(">>> ATTENTION: chosen cost function: ", environment.cost_function)

    if fast:
        SwingupPlant.ACTION_DELAY = 0.025
        environment.lookback = 13
    else:
        environment.lookback = 6

    environment.plant = GeneralizedCartpolePlant(
        hostname="192.168.2.170",
        hilscher_port="5555",
        sway_start=False,
        randomize_set_points=True,
        cost_function=SwingupPlant.cost_func_wrapper(
            environment.cost_function,
            environment.state_channels))

    environment.plot_history = plot_swingup_state_history
    environment.trajectory_plot_exploration = CartPoleTrajectoryPlot(
        cart_position_idx=0,
        cart_velocity_idx=1,
        pole_sine_idx=2,
        pole_cosine_idx=3,
        pole_velocity_idx=4)
    environment.trajectory_plot_evaluation = CartPoleTrajectoryPlot(
        cart_position_idx=0,
        cart_velocity_idx=1,
        pole_sine_idx=2,
        pole_cosine_idx=3,
        pole_velocity_idx=4)

    return environment




