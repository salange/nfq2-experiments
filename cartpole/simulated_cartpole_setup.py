import sys
import numpy as np
from typing import Callable, List, Type
from pprint import pprint

from psipy.rl.core.plant import State
from psipy.rl.core.experimentation import Environment

from psipy.rl.visualization.cartpole_plot import CartPoleTrajectoryPlot

from psipy.rl.plants.simulated.cartpole import (
    CartPoleBangAction,
    CartPoleExtendedBangAction,
    CartPole,
    CartPoleState,
    plot_swingup_state_history
)


def make_cosine_cost_func(x_boundary: float=2.4,
                          position_idx: int=0,
                          cosine_idx: int=3) -> Callable[[np.ndarray], np.ndarray]:
    def cosine_costfunc(states: np.ndarray) -> np.ndarray:
        
        position = states[:, position_idx]
        cosine = states[:, cosine_idx]

        costs = (1.0-(cosine+1.0)/2.0) / 100.0  
        costs[abs(position) >= x_boundary*0.9] = 0.1
        costs[abs(position) >= x_boundary] = 1.0

        #print(costs)

        return costs
    return cosine_costfunc


def make_sparse_cost_func(position_idx: int=0,
                          cosine_idx: int=3,
                          x_boundary: float=2.4,
                          step_cost: float=0.01,
                          use_cosine: bool=True,
                          use_upright_margin: bool=False,
                          upright_margin: float=0.3,
                          xminus: bool=True) -> Callable[[np.ndarray], np.ndarray]:
    def sparse_costfunc(states: np.ndarray) -> np.ndarray:

        position = states[:, position_idx]
        cosine = states[:, cosine_idx]

        if isinstance(cosine, np.ndarray):
            costs = np.zeros(cosine.shape)
        else:
            costs = 0.0

        if use_cosine:
            costs = (1.0-(cosine+1.0)/2.0) * step_cost  # can only get lower costs in center of x axis

        if use_upright_margin:
            costs[abs(1.0-(cosine+1.0)/2.0) > upright_margin] = step_cost

        costs[abs(position) >= x_boundary*0.2] = step_cost       # standard step costs

        if xminus: # non-terminal bad area
            costs[abs(position) >= x_boundary*0.9] = step_cost * 5  # 10x step costs close to x_boundary
        
        costs[abs(position) >= x_boundary] = 1.0                 # 100x step costs in negativ terminal states

        # ATTENTION, a word regarding the choice of terminal costs and "step
        # costs":
        # the relation of terminal costs to step costs depends on the gamma
        # value. with gamma=0.98, the geometric sequence  sum(0.98^n) converges
        # to 50 with n going to infinity (infinite lookahead), thus 100x times
        # the cost of an indiviual step seems reasonable and twice as much, as
        # the discounted future step costs can cause (50x the step cost). for
        # higher gammas closer to one, the terminal costs should be higher, to
        # prevent a terminal state's costs being lower than continuing to acting
        # within the "bounds" (aka non-terminal states). If you see your agent
        # learning to leave the bounds as quickly as possible, its likely that
        # your terminal costs are too low or your treatment of the terminal
        # transition is not correct (e.g. not doing a TD update on these
        # transitions at all, wrong scaling, etc.). We have seen both types of
        # errors in our own code as well as our students code, but also in
        # public projects and papers. So, make sure to check this twice.

        return costs
    return sparse_costfunc



def make_sway_killer_cost_func(position_idx: int=0,
                             cosine_idx: int=3,
                             x_boundary: float=2.4,
                             step_cost: float=0.01,
                             use_cosine: bool=True,
                             use_down_margin: bool=True,
                             down_margin: float=0.1,
                             xminus: bool=False) -> Callable[[np.ndarray], np.ndarray]:
    def sparse_costfunc(states: np.ndarray) -> np.ndarray:

        position = states[:, position_idx]
        cosine = states[:, cosine_idx]

        if isinstance(cosine, np.ndarray):
            costs = np.zeros(cosine.shape)
        else:
            costs = 0.0

        if use_cosine:
            costs = (cosine+1.0)/2.0 * step_cost  # shaping of costs in goal area to reward low pole angle deviations from upright position
        
        if use_down_margin:
            costs[(cosine+1.0)/2.0 > down_margin] = step_cost    # goal area is restricted to upright pole angle

        costs[abs(position) >= x_boundary*0.1] = step_cost          # standard step costs in case cart is outside center area

        if xminus:  # non-terminal bad area
            costs[abs(position) >= x_boundary*0.9] = step_cost * 5  # 5x step costs close to x_boundary
        
        # terminal bad area
        costs[abs(position) >= x_boundary] = 1.0                    # 100x step costs in terminal states
    
        return costs

    return sparse_costfunc



def setup_simulated_cartpole(environment: Environment,
                             use_extended_action_space: bool=False,
                             do_not_reset_plant: bool=False,
                             sway_killer: bool=False):

    if use_extended_action_space:
        environment.action_type = CartPoleExtendedBangAction
    else:
        environment.action_type = CartPoleBangAction

    environment.state_type  = CartPoleState

    environment.state_channels = [
        "cart_position",
        "cart_velocity",
        # "pole_theta"
        "pole_sine",
        "pole_cosine",
        "pole_velocity",
        # "move_ACT",  # add, if lookback > 1
    ]
    environment.controller_action_channels = ["move"]
    environment.batch_action_channels = ["move_index"]
    environment.lookback = 1
    environment.max_episode_length = 400
    environment.x_threshold = 3.6

    if sway_killer:
        environment.cost_function = make_sway_killer_cost_func(
            position_idx=environment.state_channels.index("cart_position"),
            cosine_idx=environment.state_channels.index("pole_cosine"),
            x_boundary=environment.x_threshold)
    else:
        environment.cost_function = make_sparse_cost_func(
            position_idx=environment.state_channels.index("cart_position"),
            cosine_idx=environment.state_channels.index("pole_cosine"),
            x_boundary=environment.x_threshold)

    print(">>> ATTENTION: chosen cost function: ", environment.cost_function)

    environment.plant = CartPole(x_threshold=environment.x_threshold,
                                 cost_function=CartPole.cost_func_wrapper(
                                     environment.cost_function,
                                     environment.state_channels),
                                 do_not_reset=do_not_reset_plant)

    environment.trajectory_plot_exploration = CartPoleTrajectoryPlot()
    environment.trajectory_plot_evaluation = CartPoleTrajectoryPlot()

    return environment



