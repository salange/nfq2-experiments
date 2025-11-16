"""Example script that learns to swing up PSIORI's version of the cartpole.
"""
import numpy as np
import sys
import os
import time
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers as tfkl
from typing import Callable, Optional
from matplotlib import pyplot as plt
from numpy import cast
from pprint import pprint
from getopt import getopt
from datetime import datetime


from psipy.rl.controllers.nfq import NFQ
from psipy.rl.controllers.nfqs import NFQs
from psipy.rl.io.batch import Batch
from psipy.rl.loop import Loop, LoopPrettyPrinter
from psipy.rl.core.experimentation import Experiment, Environment
from psipy.rl.util.schedule import LinearSchedule

from psipy.rl.visualization.plotting_callback import PlottingCallback
from psipy.rl.visualization.metrics import RLMetricsPlot
from psipy.rl.visualization.cartpole_plot import CartPoleTrajectoryPlot

from helpers import plot_last_episode
from helpers import Evaluation
from helpers import load_and_adapt_controller

from simulated_cartpole_setup import setup_simulated_cartpole
from pact_cartpole_setup import setup_pact_cartpole
from pact_cartpole_setup import setup_pact_antisway
# Define where we want to save our SART files
sart_folder = "psidata-cartpole-swingup"

# Create a model based on state, action shapes and lookback
def make_model(n_inputs, n_outputs, lookback):
    inp = tfkl.Input((n_inputs, lookback), name="state")
    net = tfkl.Flatten()(inp)
    net = tfkl.Dense(256, activation="relu")(net) # 256
    net = tfkl.Dense(256, activation="relu")(net) # 256
    net = tfkl.Dense(100, activation="tanh", name="features")(net) # 100 tanh
    net = tfkl.Dense(n_outputs, activation="sigmoid")(net) # sigmoid
    return tf.keras.Model(inp, net)

def make_nfqs_model(n_inputs, lookback):
    inp = tfkl.Input((n_inputs, lookback), name="states")
    act = tfkl.Input((1,), name="actions")
    net = tfkl.Flatten()(inp)
    net = tfkl.concatenate([act, net])
    net = tfkl.Dense(256, activation="relu")(net) # 256
    net = tfkl.Dense(256, activation="relu")(net) # 256
    net = tfkl.Dense(100, activation="tanh", name="features")(net) # 100 tanh
    net = tfkl.Dense(1, activation="sigmoid")(net) # sigmoid
    return tf.keras.Model([inp, act], net)

gamma = 0.98  # 0.98

callback = PlottingCallback(
    ax1="q", is_ax1=lambda x: x.endswith("q"), ax2="ME", is_ax2=lambda x: x.endswith("qdelta")
)

start_time = time.time()
num_cycles_rand_start = 0
fig = None
do_eval = True

def make_setup_callback(setup_controller, 
                        environment):
    def setup():
        if setup_controller is None:
            return
    
        setup_loop = Loop(environment.plant, 
                          setup_controller, 
                          "Cartpole",
                          render=environment.plant.renderable)

        setup_loop.run_episode(0, max_steps=100)
        
    return setup


def initial_fit(environment,
                controller,
                experiment: Experiment,
                setup_controller=None,
                td_iterations=200, 
                epochs_per_iteration=2,
                minibatch_size=2048,
                callback=None,
                verbose=True,
                final_fit=False,
                replay_episodes=False,
                eval_every_n_td_steps=-1,
                render=True):

    sart_folder = experiment.sart_train

    # Load the collected data
    batch = Batch.from_hdf5(
        sart_folder,
        state_channels=environment.state_channels,
        action_channels=environment.batch_action_channels,
        lookback=environment.lookback,
        control=controller,
    )

    print("Initial fitting with {} episodes from {} for {} iterations with {} epochs each and minibatch size of {}.".format(len(batch._episodes),   sart_folder, td_iterations, epochs_per_iteration, minibatch_size))

    # fakes = create_fake_episodes(sart_folder, lookback, batch.num_samples)
    # batch.append(fakes)

    print("Initial fitting with data from {} for {} iterations with {} epochs each and minibatch size of {}.".format(sart_folder, td_iterations, epochs_per_iteration, minibatch_size))


    callbacks = [callback] if callback is not None else None

    evaluation = None
    metrics_plot = None
    state_eval_figure = None
    if eval_every_n_td_steps > 0:
        evaluation = Evaluation(environment=environment,
                                controller=controller,
                                sart_folder=experiment.sart_eval,
                                loop_identifier="Cartpole-Evaluation-Initial-Fit",
                                render=render and environment.plant.renderable,
                                min_avg_step_cost_init=0.1,
                                setup_callback=make_setup_callback(setup_controller, environment))
        metrics_plot = RLMetricsPlot(filename=experiment.plot_folder + "/initial-fit-metrics-latest.png")

    if replay_episodes:
        print(">>>> Replaying {} episodes".format(len(batch._episodes)))
       
        try:
            for i in range(1, len(batch._episodes)):  
                print("Replaying episode:", i)

                replay_batch = Batch(episodes=batch._episodes[0:i],         
                                     control=controller)
                                     
                #pprint (replay_batch._episodes[0].observations)
                #sys.exit()

                if i == 1 or (i % 10 == 0 and i < len(batch._episodes) / 2):
                    print("Fitting normalizer...")
                    controller.fit_normalizer(replay_batch.observations, method="meanstd")
            

                controller.fit(
                        replay_batch,
                        costfunc=environment.cost_function,
                        iterations=4, # TODO: parameters
                        epochs= 8, # TODO: parameters
                        minibatch_size=minibatch_size,
                        gamma=gamma,
                        callbacks=[callback],
                        verbose=verbose,
                )
        except KeyboardInterrupt:
            pass

    else:
    
        print("Fitting normalizer...")
        controller.fit_normalizer(batch.observations, method="meanstd")
    
        # Fit the controller
        print("Initial fitting of controller...")

        iterations = 0
        td_steps = td_iterations
        episode = 0  # count for evaluations

        if eval_every_n_td_steps > 0:
            td_steps = eval_every_n_td_steps

        while iterations < td_iterations:
            try:
                controller.fit(
                    batch,
                    costfunc=environment.cost_function,
                    iterations=td_steps,
                    epochs=epochs_per_iteration,
                    minibatch_size=minibatch_size,
                    gamma=gamma,
                    callbacks=callbacks,
                    verbose=verbose)
            except KeyboardInterrupt:
                pass

            iterations += td_steps  # update counter even if interrupted

            if eval_every_n_td_steps > 0:
                avg_step_cost, reps, episode_metrics = evaluation.evaluate(
                    episode=episode,
                    max_episode_length=environment.max_episode_length,
                    min_eps_before_eval=10,
                    default_repetitions=1, 
                    additional_repetitions=0
                )

                #print(">>> metrics['avg_cost']", metrics["avg_cost"])
                print(">>> episode_metrics", episode_metrics)
            
                with open(experiment.folder + '/initial-fit-metrics-latest', 'wt') as out:
                    pprint(evaluation._metrics, stream=out)
            
                pprint(evaluation.transposed_average_metrics())
                metrics_plot.update(evaluation.transposed_average_metrics())
                metrics_plot.plot()

                if metrics_plot.filename is not None:
                    print(">>>>>>>> SAVING PLOT <<<<<<<<<<<")
                    metrics_plot.save()

                if avg_step_cost < evaluation._min_avg_step_cost * 1.075:
                    filename = experiment.folder + "/model-candidate-initial-fit-{}-avg_cost-{}".format(episode, str(avg_step_cost).replace(".", "_"))
                    print("Saving candidate model: ", filename)
                    controller.save(filename)
                           
                if avg_step_cost == evaluation._min_avg_step_cost:
                    print("Saving very best model")
                    try:
                        os.rename(experiment.folder + "/model-initial-fit-very_best.zip",
                                  experiment.folder + "/model-initial-fit-second_best.zip")
                    except OSError:
                        pass

                    controller.save(experiment.folder + "/model-initial-fit-very_best")

                # also plot the evaluation

                state_eval_figure = plot_last_episode(
                    environment.trajectory_plot_evaluation,
                    experiment.sart_eval,
                    environment,
                    figure=state_eval_figure,
                    filename=  f"{ experiment.plot_folder }/eval_initial_fit_episode-{ episode }.png",
                    episode_num=episode,
                    do_display=True,
                    title_string="Evaluation"
                )

            episode += 1

    try:
        if final_fit:
                        # Fit the controller
            controller.fit(
                batch,
                costfunc=environment.cost_function,
                iterations=25, # iterations,
                epochs= 8,
                minibatch_size=minibatch_size, #batch_size,
                gamma=gamma,
                callbacks=[callback],
                verbose=verbose,
            )

    except KeyboardInterrupt:
        pass

    controller.save(experiment.folder + "/model-initial-fit")    

def learn(environment, 
          controller, 
          experiment: Experiment,
          setup_controller=None,
          num_episodes=-1,
          max_episode_length=400,
          refit_normalizer=False, 
          do_eval=True):
          
    state_figure = None
    state_eval_figure = None

    metrics = { "total_cost": [], "avg_cost": [], "cycles_run": [], "wall_time_s": [] }
    min_avg_step_cost = 0.01    # only if avg costs of an episode are less than 100+x% of this, we potentially save the model (must be near the best)
    min_eps_before_eval = 10    

    sart_folder = experiment.sart_train
    sart_folder_eval = experiment.sart_eval

    episode = 0

    epsilon_schedule = LinearSchedule(start=0.8,
                                      end=0.05,  # 0.1
                                      num_episodes=num_episodes / 4)
  
    setup_callback = make_setup_callback(setup_controller, environment)

    eval_reps = 1     
    evaluation = Evaluation(environment=environment,
                            controller=controller,
                            sart_folder=sart_folder_eval,
                            loop_identifier="Cartpole-Evaluation",
                            render=False,
                            min_avg_step_cost_init=0.1,
                            setup_callback=setup_callback)                               
    
    try:
        batch = Batch.from_hdf5(
            sart_folder,
            state_channels=environment.state_channels,
            action_channels=environment.batch_action_channels,
            lookback=environment.lookback,
            control=controller,
        )
        print(f"Found {len(batch._episodes)} episodes in {sart_folder}. Will use these for fitting and continue with episode {len(batch._episodes)}")

        if refit_normalizer:
            print("Refit the normalizer again using meanstd.")
            controller.fit_normalizer(batch.observations, method="meanstd")

        episode = len(batch._episodes)
        

    except OSError:
        print("No saved episodes found, starting from scratch.")

    metrics_plot = RLMetricsPlot(filename=experiment.plot_folder + "/metrics-latest.png")

    pp = LoopPrettyPrinter(environment.cost_function)

    loop = Loop(environment.plant, controller, "Cartpole", sart_folder, render=environment.plant.renderable)



    while episode < num_episodes or num_episodes < 0:
        print("Starting episode:", episode)

        controller.epsilon = epsilon_schedule.value(episode)
        print("NFQ Epsilon:", controller.epsilon)

        setup_callback()
        loop.run_episode(episode, max_steps=max_episode_length, pretty_printer=pp)


        state_figure = plot_last_episode(
            environment.trajectory_plot_exploration,
            sart_folder,
            environment,
            figure=state_figure,
            filename=  f"{ experiment.plot_folder }/episode-{ episode }.png",
            episode_num=episode
        )

        # Load the collected data
        batch = Batch.from_hdf5(
            sart_folder,
            state_channels=environment.state_channels,
            action_channels=environment.batch_action_channels,
            lookback=environment.lookback,
            control=controller,
        )

        if refit_normalizer and episode % 10 == 0 and episode < num_episodes / 2:   
            print("Refit the normalizer again using meanstd.")
            controller.fit_normalizer(batch.observations, method="meanstd")


        try:
            # Fit the controller
            controller.fit(
            batch,
            costfunc=environment.cost_function,
            iterations=4, # iterations,
            epochs=8,
            minibatch_size=2048, #batch_size,
            gamma=gamma,
            callbacks=[callback],
            verbose=1,
        )
        except KeyboardInterrupt:
            pass


        try:
            os.rename(experiment.folder + "/model-latest.zip", 
                      experiment.folder + "/model-latest-backup.zip")
        except OSError:
            pass
        controller.save(experiment.folder + "/model-latest")  # this is always saved to allow to continue training after
    
    
        if do_eval:
            avg_step_cost, reps, episode_metrics = evaluation.evaluate(
                episode,
                max_episode_length=max_episode_length,
                min_eps_before_eval=10,
                default_repetitions=1, 
                additional_repetitions=0
            )

            #print(">>> metrics['avg_cost']", metrics["avg_cost"])
            print(">>> episode_metrics", episode_metrics)
            
            with open(experiment.folder + '/metrics-latest', 'wt') as out:
                pprint(evaluation._metrics, stream=out)
            
            pprint(evaluation.transposed_average_metrics())
            metrics_plot.update(evaluation.transposed_average_metrics())
            metrics_plot.plot()

            if metrics_plot.filename is not None:
                print(">>>>>>>> SAVING PLOT <<<<<<<<<<<")
                metrics_plot.save()

            if avg_step_cost < evaluation._min_avg_step_cost * 1.075:
                filename = experiment.folder + "/model-candidate-{}-avg_cost-{}".format(len(batch._episodes), str(avg_step_cost).replace(".", "_"))
                print("Saving candidate model: ", filename)
                controller.save(filename)
                           
            if avg_step_cost == evaluation._min_avg_step_cost:
                print("Saving very best model")
                try:
                    os.rename(experiment.folder + "/model-very_best.zip", 
                              experiment.folder + "/model-second_best.zip")
                except OSError:
                    pass

                controller.save(experiment.folder + "/model-very_best")

            # also plot the evaluation

            state_eval_figure = plot_last_episode(
                environment.trajectory_plot_evaluation,
                sart_folder_eval,
                environment,
                figure=state_eval_figure,
                filename=  f"{ experiment.plot_folder }/eval_episode-{ len(batch._episodes) }.png",
                episode_num=len(batch._episodes),
                do_display=False,
                title_string="Evaluation"
            )

        episode += 1

    print("Elapsed time:", time.time() - start_time)


def play(plant,
         controller,
         setup_controller=None,
         sart_folder="psidata-cartpole-play",
         num_episodes=-1):
    
    setup_callback = make_setup_callback(setup_controller,
                                         environment)
    episode = 0
    loop = Loop(plant, controller, "Hardware Swingup", sart_folder, render=plant.renderable)

    while episode < num_episodes or num_episodes < 0:
        setup_callback()
        loop.run_episode(episode, max_steps=-1);
        episode += 1



if __name__ == "__main__":
    start_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    load_network = False
    do_initial_fit = False
    play_only = False
    controller = None
    setup_controller = None
    experiment_folder = "experiment-" + start_time_str
    play_after_initial_fit = False
    refit_normalization = True
    model_filename = None
    type_of_plant = "simulated"
    use_extended_action_space = False
    num_episodes = 500
    use_nfqs = False
    setup_model_filename = None
    use_sway_killer = False
    experiment_type = None

    try:
        opts, args = getopt(sys.argv[1:], "hfpe:l:r:t:xn:",
                            ["help", "play-only", "initial-fit", "experiment-folder=", "load-model=", "refit=", "plant-type=", "load-setup-model=", "num-episodes=", "extended-action-space", "nfqs", "sway-killer", "experiment-type="])
    except getopt.GetoptError as err:
        print("Usage: python nfq_hardware_swingup.py [--play <model.zip>]")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print("Usage: python nfq_hardware_swingup.py [--play <model.zip>]")
            sys.exit()
        elif opt in ("-p", "--play-only"):
            play_only = True
        elif opt in ("-f", "--initial-fit"):
            do_initial_fit = True
        elif opt in ("-e", "--experiment-folder"):
            experiment_folder = arg
        elif opt in ("-l", "--load-model"):
            print("Loading controller from file with name: ", arg)
            model_filename = arg
        elif opt in ("-r", "--refit"):
            refit_normalization = arg.lower() == "true"
        elif opt in ("-t", "--plant-type"):
            if arg not in ("simulated", "real"):
                raise ValueError("Plant type must be either 'simulated' or 'real'")
            type_of_plant = arg
        elif opt in ("-x", "--extended-action-space"):
            use_extended_action_space = True
        elif opt in ("--sway-killer"):
            use_sway_killer = True
        elif opt in ("-n", "--num-episodes"):   
            num_episodes = int(arg)
        elif opt in ("--nfqs"):
            use_nfqs = True
        elif opt in ("--load-setup-model"):
            setup_model_filename = arg
        elif opt in ("--experiment-type"):
            experiment_type = arg

    environment = Environment()
    experiment = Experiment()

    experiment.folder = experiment_folder

    # Create the experiment folder if it does not exist
    os.makedirs(experiment_folder, exist_ok=True)

    experiment.sart_train = experiment_folder + "/sart-train"
    experiment.sart_eval = experiment_folder + "/sart-eval"
    experiment.sart_play = experiment_folder + "/sart-play"
    experiment.sart_initial_fit_eval = experiment_folder + "/sart-initial-fit-eval"
    experiment.plot_folder = experiment_folder + "/plots"

    os.makedirs(experiment.plot_folder, exist_ok=True)

    do_not_reset_plant = setup_model_filename is not None

    if type_of_plant == "simulated":    
        setup_simulated_cartpole(environment,
                                 use_extended_action_space,
                                 do_not_reset_plant=do_not_reset_plant,
                                 sway_killer=use_sway_killer)
    else:
        if experiment_type is None or experiment_type == "swingup":
            setup_pact_cartpole(environment, 
                                use_extended_action_space=use_extended_action_space, 
                                sway_killer=use_sway_killer)
        elif experiment_type == "antisway":
            setup_pact_antisway(environment, 
                                use_extended_action_space=use_extended_action_space)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}") 

    if use_nfqs:
       environment.batch_action_channels = environment.controller_action_channels

    if model_filename is not None:
        controller = load_and_adapt_controller(model_filename, 
                                               environment,
                                               use_nfqs=use_nfqs)
        print("ACTION TYPE IN CONTROLLER: ", controller.action_type)


    if setup_model_filename is not None:
        setup_controller = NFQ.load(setup_model_filename)
        
    if controller is None:  # not loaded from file
        if use_nfqs:
            print("Creating a new NFQs controller.")

            #environment.batch_action_channels = environment.controller_action_channels

            model = make_nfqs_model(len(environment.state_channels), 
                                    environment.lookback)


            controller = NFQs(
                model=model,
                state_channels=environment.state_channels,
                action_channels=environment.controller_action_channels,
                action=environment.action_type,
                action_values=environment.action_type.legal_values[0],
                optimizer=keras.optimizers.Adam(),
                lookback=environment.lookback,
                scale=False)

        else:
            print("Creating a new NFQ controller.")

            model = make_model(len(environment.state_channels), 
                               len(environment.action_type.legal_values[0]), 
                               environment.lookback)

            controller = NFQ(
                model=model,
                state_channels=environment.state_channels,
                action_channels=environment.controller_action_channels,
                action=environment.action_type,
                action_values=environment.action_type.legal_values[0],
                optimizer=keras.optimizers.Adam(),
                lookback=environment.lookback,
                scale=False) # output scaling
        
            
    experiment.serialize(experiment.folder + "/experiment" + start_time_str + ".yaml")
    #environment.serialize(experiment.folder + "/environment" + start_time_str + ".yaml")

    if do_initial_fit:
        initial_fit(environment, 
                    controller,
                    experiment=experiment,
                    setup_controller=setup_controller,
                    callback=callback,
                    td_iterations=400,
                    epochs_per_iteration=4,
                    eval_every_n_td_steps=8)

    # CREATE REAL PLANT HERE BECAUSE CONNECTION LOSS?

    if play_only:
        play(environment.plant, 
             controller, 
             setup_controller,
             sart_folder=experiment.sart_play)
        sys.exit()

    else:
        learn(environment, 
              controller, 
              experiment=experiment,
              setup_controller=setup_controller,
              num_episodes=num_episodes,
              max_episode_length=environment.max_episode_length,
              refit_normalizer=refit_normalization)



