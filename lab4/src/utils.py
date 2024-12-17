import time
import random
import numpy as np
import dataclasses
import wandb
import time
import torch
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import List, Tuple, Union, Any, Dict, Optional


StateT = Union[int, float, Tuple[Union[float, int]]]
ActionT = Any


@dataclass
class TrainConfig:
    agent: str
    results_path: str
    mcvi_exprob: float
    mcvi_episodes: int
    tabular_exprob: float
    tabular_episodes: int
    # [Optional] Policy Gradient Training
    is_resume: bool
    pretrain_path: str
    batch_size: int
    num_updates: int
    max_t: int
    gamma: float
    lr: float
    save_every: int
    window_size: int
    track: bool
    # Only work when track is True:
    group: str
    project: str
    run_id: str
    entity: str
    wandb_dir: str


def create_bins(low: List[float], high: List[float], num_bins: Union[int, List[int]]) -> List[np.ndarray]:
    """
    Takes in a gym.spaces.Box and returns a set of bins per feature according to num_bins
    """
    assert len(low) == len(high)
    if isinstance(num_bins, int):
        num_bins = [num_bins for _ in range(len(low))]
    assert len(num_bins) == len(low)
    bins = []
    for low, high, n in zip(low, high, num_bins):
        bins.append(np.linspace(low, high, n))
    return bins


def discretize(x, bins) -> Tuple[int]:
    """
    Discretize an array x according to bins
    x: np.ndarray, shape (features,)
    bins: np.ndarray, shape (features, bins)
    """
    return tuple(int(np.digitize(feature, bin)) for feature, bin in zip(x, bins))


# An abstract class representing a Markov Decision Process (MDP).
class MDP:
    # Return the start state.
    def startState(self):
        raise NotImplementedError("Override me")

    # Property holding the set of possible actions at each state.
    @property
    def actions(self) -> List[ActionT]:
        raise NotImplementedError("Override me")

    # Property holding the discount factor
    @property
    def discount(self):
        raise NotImplementedError("Override me")

    # property holding the maximum number of steps for running the simulation.
    @property
    def time_limit(self) -> int:
        raise NotImplementedError("Override me")

    # Transitions the MDP
    def transition(self, action):
        raise NotImplementedError("Override me")


class RLAlgorithm:
    """
    Abstract class:
        An RLAlgorithm performs reinforcement learning.  All it needsto know is the
        set of available actions to take.  The simulator (see simulate()) will call
        getAction() to get an action, perform the action, and then provide feedback
        (via incorporateFeedback()) to the RL algorithm, so it can adjust its parameters.
    """

    # Your algorithm will be asked to produce an action given a state.
    def getAction(self, state: StateT) -> ActionT:
        raise NotImplementedError("Override me")

    # We will call this function when simulating an MDP, and you should update
    # parameters.
    # If |state| is a terminal state, this function will be called with (s, a,
    # 0, None). When this function is called, it indicates that taking action
    # |action| in state |state| resulted in reward |reward| and a transition to state
    # |nextState|.
    def incorporateFeedback(self, state: StateT, action: ActionT, reward: int, nextState: StateT, terminal: bool):
        raise NotImplementedError("Override me")


# Class for untrained agent which takes random action every step.
# This class is used as a benchmark at the start of the assignment.
class RandomAgent(RLAlgorithm):
    def __init__(self, actions: List[ActionT]):
        self.actions = actions

    def getAction(self, state: StateT, explore: bool = False):
        return random.choice(self.actions)

    def incorporateFeedback(self, state: StateT, action: ActionT, reward: int, nextState: StateT, terminal: bool):
        pass


def simulate(mdp: MDP, rl: RLAlgorithm, numTrials=10, train=True, verbose=False, demo=False):
    """
    Perform |numTrials| of the following:
        On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulates the
        RL algorithm according to the dynamics of the MDP.
        Return the list of rewards that we get for each trial.
    """
    totalRewards = []  # The discounted rewards we get on each trial
    for trial in range(numTrials):
        state = mdp.startState()
        if demo:
            mdp.env.render()
        totalDiscount = 1
        totalReward = 0
        trialLength = 0
        for _ in range(mdp.time_limit):
            if demo:
                time.sleep(0.05)
            action = rl.getAction(state, explore=train)
            if action is None:
                break
            nextState, reward, terminal = mdp.transition(action)
            trialLength += 1
            if train:
                rl.incorporateFeedback(state, action, reward, nextState, terminal)

            totalReward += totalDiscount * reward
            totalDiscount *= mdp.discount
            state = nextState

            if terminal:
                break  # We have reached a terminal state

        if verbose and trial % 100 == 0:
            print(("Trial %d (totalReward = %s, Length = %s)" % (trial, totalReward, trialLength)))
        totalRewards.append(totalReward)
    return totalRewards


def init_config_from_args(cls, args):
    """Initialize a dataclass from a Namespace."""
    return cls(**{f.name: getattr(args, f.name) for f in dataclasses.fields(cls)})


def get_date_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def handle_results_path(res_path: str, default_root: str = "../results") -> Path:
    """Sets results path if it doesn't exist yet."""
    if res_path is None:
        results_path = Path(default_root)
    else:
        results_path = Path(res_path)
    print(f"Results will be saved to '{results_path}'")
    return results_path


# [Optional] For Tracking Experiments
class WeightsBiasesTracker:
    """
    A class to handle all logging to Weights & Biases.

    Attributes:
        run_id: str, the unique identifier for the run.
        run_dir: Path, the directory where the logs will be saved.
        hparams: Dict[str, Any], the hyperparameters for the run.
        project: str, the name of the project.
        entity: Optional[str], the name of the entity.
        group: str, the name of the group.

    Methods:
        initialize: Initialize the W&B run.
        write_hyperparameters: Write the hyperparameters to the run.
        write: Write metrics to the run.
        finalize: Finish the run.
    """

    def __init__(
        self,
        run_id: str,
        run_dir: Path,
        hparams: Dict[str, Any],
        project: str = "CartPole",
        entity: Optional[str] = None,
        group: str = None,
    ) -> None:
        self.run_id, self.run_dir, self.hparams = run_id, run_dir, hparams

        # Get W&B-Specific Initialization Parameters
        self.project, self.entity, self.group, self.wandb_dir = project, entity, group, self.run_dir

        # Call W&B.init()
        self.initialize()

    def initialize(self) -> None:
        wandb.init(
            name=self.run_id,
            dir=self.wandb_dir,
            config=self.hparams,
            project=self.project,
            entity=self.entity,
            group=self.group,
        )

    def write_hyperparameters(self) -> None:
        wandb.config = self.hparams

    def write(self, global_step: int, metrics: Dict[str, Union[int, float]]) -> None:
        wandb.log(metrics, step=global_step)

    @staticmethod
    def finalize() -> None:
        wandb.finish()
        # A job gets 10 seconds to get its affairs in order
        time.sleep(10)


class Metrics:
    """
    A class to handle all logging to Weights & Biases.

    Attributes:
        run_id: str, the unique identifier for the run.
        run_dir: Path, the directory where the logs will be saved.
        hparams: Dict[str, Any], the hyperparameters for the run.
        wandb_project: str, the name of the project.
        wandb_entity: Optional[str], the name of the entity.
        window_size: int, the size of the window for smoothing.

    Methods:
        log: Log metrics for the current step.
        get_status: Get the status of the current step.
        commit: Update the metrics for the current step.
        push: Push the metrics to the trackers.
        finalize: Finalize the trackers.
    """

    def __init__(
        self,
        stage: str,
        run_id: str,
        run_dir: Path,
        hparams: Dict[str, Any],
        wandb_project: str = "CartPole",
        wandb_entity: Optional[str] = None,
        window_size: int = 128,
    ) -> None:
        self.run_id, self.run_dir, self.hparams, self.stage = run_id, run_dir, hparams, stage

        # Initialize Tracker
        tracker = WeightsBiasesTracker(run_id, run_dir, hparams, project=wandb_project, entity=wandb_entity)

        # Add Hyperparameters --> add to `self.trackers`
        tracker.write_hyperparameters()
        self.tracker = tracker

        # Create Universal Metrics Buffers
        self.step, self.start_time, self.step_start_time = 0, time.time(), time.time()
        self.state = {
            "loss": deque(maxlen=window_size),
            "reward": deque(maxlen=window_size),
            "step_time": deque(maxlen=window_size),
            "lr": [],
        }

    def log(self, step: int, metrics: Dict[str, Union[int, float]]) -> None:
        self.tracker.write(step, metrics)

    def get_status(self, loss: Optional[torch.Tensor] = None) -> str:
        reward = np.mean(list(self.state["reward"]))
        if loss is None:
            return f"=>> [Step] {self.step:06d} =>> Reward :: {reward:.2f}"

        # Otherwise, embed `loss` in status report!
        return f"=>> [Step] {self.step:06d} =>> Reward :: {reward:.2f} -- Loss :: {loss:.4f}"

    def commit(
        self, *, step: Optional[int] = None, lr: Optional[float] = None, update_step_time: bool = False, **kwargs
    ) -> None:
        """Update all metrics in `self.state` by iterating through special positional arguments & kwargs."""
        if step is not None:
            self.step = step

        # Special Positional Arguments
        if lr is not None:
            self.state["lr"].append(lr)

        if update_step_time:
            self.state["step_time"].append(time.time() - self.step_start_time)
            self.step_start_time = time.time()

        # Generic Keyword Arguments
        for key, value in kwargs.items():
            if key == "loss":
                loss_val = value.detach()
                self.state["loss"].append(loss_val)
            else:
                self.state[key].append(value)

    def push(self) -> str:
        """Push the metrics to the trackers."""
        loss = torch.stack(list(self.state["loss"])).mean().item()
        reward, step_time, lr = (
            np.mean(list(self.state["reward"])),
            np.mean(list(self.state["step_time"])),
            self.state["lr"][-1],
        )
        status = self.get_status(loss)

        # Fire to Trackers
        prefix = self.stage.capitalize()
        self.log(
            self.step,
            metrics={
                f"{prefix}/Step": self.step,
                f"{prefix}/Loss": loss,
                f"{prefix}/Reward": reward,
                f"{prefix}/Learning Rate": lr,
                f"{prefix}/Step Time": step_time,
            },
        )
        return status

    def finalize(self) -> str:
        self.tracker.finalize()
