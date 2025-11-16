import sys
import numpy as np

from psipy.rl.io.batch import Batch
from psipy.rl.core.controller import Controller
from psipy.rl.loop import Loop
from psipy.rl.core.experimentation import Environment
from psipy.rl.controllers import NFQ, NFQs


def load_and_adapt_controller(model_filename: str, 
                              environment: Environment,
                              use_nfqs=False):
    
    if use_nfqs:
        controller = NFQs.load(model_filename,
                               custom_objects=[environment.action_type])
     

        if not controller.action_type == environment.action_type or not len(controller.action_values) == len(environment.action_type.legal_values[0]):
           # the following relies on knowledge of the internal implementation of nfqs. Thus, might break
           # in the future. Ability to change actions should better go to the controller itself.

           print ("ATTENTION: CHANGING ACTIONS IN NFQ NETWORK. THIS MIGHT BREAK THIS NETWORK / DATA IN CASE OLD AND NEW ACTIONS ARE NOT A UPERSET OF THE OTHER.")

           controller.action_type = environment.action_type
           controller.action_values = environment.action_type.legal_values[0]
           controller.action_values_normalized = controller.action_normalizer.transform(controller.action_values[..., None]).flatten()

           print ("ACTION VALUES:", controller.action_values)
           print ("NORMALIZED ACTION VALUES:", controller.action_values_normalized)
        
    else:
        controller = NFQ.load(model_filename,
                              custom_objects=[environment.action_type])

        print (controller._model.output)
        print (controller._model.output.shape)

        if (controller._model.output.shape[1] != len(environment.action_type.legal_values[0])):
           print ("ATTENTION: Reshape model output to match action space.")
           controller = controller.model_with_extended_actions(environment.action_type)

           print (controller._model.output)
           print (controller._model.output.shape)

    return controller


def plot_last_episode(trajectory_plot,
                      sart_folder,
                      environment: Environment,
                      title_string=None,
                      filename=None,
                      episode_num=None,
                      do_display=True,
                      figure=None):
    """ Load the last episode from a sart file and plot it."""

    last_episode_internal = Batch.from_hdf5(  # load plant-internal channels
        sart_folder,
        action_channels=environment.batch_action_channels,
        lookback=environment.lookback, 
        only_newest=1
    )

    if last_episode_internal is None:
        print("Last episode was too short to plot. Skipping it.")
        return 

    trajectory_plot.update(last_episode_internal._episodes[0],
                           episode_num=episode_num,
                           title_string=title_string)
    if do_display:
        trajectory_plot.plot()
    if filename:
        trajectory_plot.save(filename=filename)


class Evaluation:
    """ Evaluates a specific controller on a specific plant. Can run several 
        evaluations over time in order to collect metrics during "training" of
        the controller."""

    def __init__(self,
                 environment: Environment,
                 controller: Controller,
                 sart_folder,
                 loop_identifier="Evaluation",
                 render=False,
                 min_avg_step_cost_init=1.0,
                 setup_callback=None
                 ) -> None:
        
        self._environment = environment
        self._controller = controller
        self._sart_folder = sart_folder
        self._setup_callback = setup_callback

        self._loop = Loop(
            self._environment.plant, 
            self._controller, 
            loop_identifier,
            self._sart_folder, 
            render=environment.plant.renderable and render)
        
        self._metrics = {} # structure: episode_num -> array of 1 to many results of evaluation episodes

        self._min_avg_step_cost = min_avg_step_cost_init
        

    def _add_metrics(self, episode, episode_metrics):
        if episode in self._metrics:
            self._metrics[episode] = (self._metrics[episode] or []).append(episode_metrics)
        else:
            self._metrics[episode] = episode_metrics

    def _calc_avg_step_cost_from_repetitions(self, episode_metrics):
        return np.sum([h["total_cost"] for h in episode_metrics]) / np.sum([h["cycles_run"] for h in episode_metrics])

    def _run_evaluations(self,
                         repetitions,
                         max_episode_length):
        old_epsilon = self._controller.epsilon  # store controller's state
        self._controller.epsilon = 0.0

        self._setup_callback()
        self._loop.run(repetitions,
                       max_episode_steps=max_episode_length)
        self._controller.epsilon = old_epsilon  # resetore controller's state

        episode_metrics = [v for k,v in self._loop.metrics.items() if k <= repetitions]  # loop will store the results in a dictionary episode => metrics_hash with the dictionary's keys being the episode number counting from 1 to repetitions! (starts counting from 1, not 0)

        return episode_metrics
    
    def metrics_of_episode(self, episode):
        if not episode in self._metrics: 
            return None
        
        return self._metrics[episode]
    
    def _average_metrics(self, episode_metrics):
        metrics = {}

        reps = len(episode_metrics) * 1.0   # make it float
        metrics["cycles_run"] = np.sum([h["cycles_run"] for h in episode_metrics]) / reps
        metrics["total_cost"] = np.sum([h["total_cost"] for h in episode_metrics]) / reps
        metrics["wall_time_s"] = np.sum([h["wall_time_s"] for h in episode_metrics]) / reps

        metrics["avg_cost"] = metrics["total_cost"] / metrics["cycles_run"]
        
        return metrics


    def average_metrics_of_episode(self, episode):
        episode_metrics = self.metrics_of_episode(episode)
        if episode_metrics is None:
            return None
        
        return self._average_metrics(episode_metrics)


    def average_metrics(self):
        metrics = {}
        for key in self._metrics.keys():
            metrics[key] = self.average_metrics_of_episode(key)

        return metrics
    

    def transposed_average_metrics(self):
        metrics = self.average_metrics()
        metrics_transposed = {}

        for key in np.sort(list(metrics.keys())):
            for metric_name, metric_value in metrics[key].items():
                if metric_name not in metrics_transposed:
                    metrics_transposed[metric_name] = []
                
                metrics_transposed[metric_name].append(metric_value)

        return metrics_transposed



    def evaluate(self,
                 episode,
                 max_episode_length=500,
                 min_eps_before_eval=10,
                 default_repetitions=1, 
                 additional_repetitions=0):

        episode_metrics = self._run_evaluations(default_repetitions,
                                                max_episode_length)
        repetitions = default_repetitions
        self._add_metrics(episode, episode_metrics)

        avg_step_cost = self._calc_avg_step_cost_from_repetitions(self._metrics[episode])

        print(">>> INITIAL EPISODE METRICS: {} with avg step costs {}".format(self._metrics[episode], avg_step_cost))

        if (episode > min_eps_before_eval and 
            avg_step_cost < self._min_avg_step_cost * 1.1 and
            additional_repetitions > 0):
                print("Running {} ADDITIONAL EVALUATION repetitions because model is promising candidate for replacing the best model found so far...".format(additional_repetitions))

                episode_metrics = self._run_evaluations(additional_repetitions,
                                                        max_episode_length)
                self._add_metrics(episode, episode_metrics)
                repetitions += additional_repetitions

                avg_step_cost = self._calc_avg_step_cost_from_repetitions(self._metrics[episode])

        if avg_step_cost < self._min_avg_step_cost:
            self._min_avg_step_cost = avg_step_cost

        return (avg_step_cost, repetitions,
                self._average_metrics(self._metrics[episode]))             
            
