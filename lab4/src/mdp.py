import gymnasium as gym
import numpy as np
from typing import Any, Optional, List
from utils import create_bins, discretize, MDP, StateT, ActionT


class MoutainCarMDP(MDP):
    """
    The Mountain Car MDP

    Attributes:
        - env: gym.Env, the environment
        - state_space: int, the dimension of the state space
        - action_space: int, the dimension of the action space
        - _discount: float, the discount factor of the MDP
        - _time_limit: int, the maximum number of steps before the MDP should be reset
        - _actions: list, the set of actions possible in every state
        - _reset_seed_gen: np.random.Generator, the random number generator for resetting the environment
        - low: List[float], the lower bounds of the state space
        - high: List[float], the upper bounds of the state space
        - bins: np.ndarray, the bins to discretize the state space

    Methods:
        - state_adapter: discretize the state space
        - startState: reset the environment and return the initial state
        - reward: return the custom reward function
        - transition: take an action in the environment
    """

    def __init__(
        self,
        discount: float = 0.99,
        time_limit: Optional[int] = None,
        num_bins: Optional[int] = 20,
        low: Optional[List[float]] = None,
        high: Optional[List[float]] = None,
        render_mode: str = None,
        seed: int = 0,
    ):
        super().__init__()
        self.env = gym.make("MountainCar-v0", render_mode=render_mode)
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self._discount = discount
        self._time_limit = time_limit
        self._actions = list(range(self.env.action_space.n))
        self._reset_seed_gen = np.random.default_rng(seed=seed)
        self.low = self.env.observation_space.low if low is None else low
        self.high = self.env.observation_space.high if high is None else high
        self.bins = create_bins(self.low, self.high, num_bins)

    # The maximum number of steps before the MDP should be reset
    @property
    def time_limit(
        self,
    ) -> int:
        return self._time_limit

    # The set of actions possible in every state
    @property
    def actions(
        self,
    ) -> list[Any]:
        return self._actions

    # The discount factor of the MDP
    @property
    def discount(
        self,
    ) -> float:
        return self._discount

    def state_adapter(self, state: StateT) -> StateT:
        return discretize(state, self.bins)

    def startState(self) -> StateT:
        """
        Reset the environment and return the initial state

        Returns:
            start_state: np.ndarray, the initial state
        """
        observation, info = self.env.reset(seed=int(self._reset_seed_gen.integers(0, 1e6)))
        return self.state_adapter(observation)

    # Returns custom reward function
    def reward(self, nextState, originalReward):
        if "MountainCar-v0" in self.env.unwrapped.spec.id:
            # reward fn based on x position and velocity
            position_reward = -(self.high[0] - nextState[0])
            velocity_reward = -(self.high[1] - np.abs(nextState[1]))
            return position_reward + velocity_reward
        else:
            return originalReward

    def transition(self, action: ActionT) -> tuple[np.ndarray, float, bool]:
        """
        Take an action in the environment

        Args:
            action: int, the action to take

        Returns:
            next_state: np.ndarray, the next state
            reward: float, the reward
            terminated: bool, whether the episode is terminated
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        reward = self.reward(next_state, reward)
        next_state = self.state_adapter(next_state)
        return next_state, reward, terminated
