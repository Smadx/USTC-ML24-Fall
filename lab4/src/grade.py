import argparse
import gymnasium as gym
import torch
import numpy as np
from typing import Tuple, Union
from tqdm import tqdm
from pathlib import Path

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

from mdp import MoutainCarMDP

from submission import Policy, ModelBasedMonteCarlo, TabularQLearning


def evaluate_policy_agent(agent: Policy, n_eval_episodes: int, seed: int = 42) -> Tuple[float, float]:
    """
    Evaluate a policy agent in the CartPole environment.

    Args:
        - agent: Policy, the policy agent to evaluate
        - n_eval_episodes: int, the number of episodes to evaluate the agent
        - seed: int, the seed for the environment

    Returns:
        - mean_reward: float, the mean reward of the agent
        - std_reward: float, the standard deviation of the reward of the agent
    """
    env = gym.make("CartPole-v1")

    num_params = sum(p.numel() for p in agent.parameters())
    num_params_k = num_params / 10**3
    print(f"Parameters : {num_params_k:.3f}K Total.")
    assert num_params_k < 100, "Your model is too large!"

    agent.eval()
    gen = np.random.default_rng(seed=seed)
    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        state, _ = env.reset(seed=int(gen.integers(0, 1e6)))
        step = 0
        done = False
        total_rewards_ep = 0

        while True:
            state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
            action, _ = agent.getAction(state)
            state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break

        episode_rewards.append(total_rewards_ep)

    env.close()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


def evaluate_value_agent(
    agent: Union[ModelBasedMonteCarlo, TabularQLearning], max_steps: int, n_eval_episodes: int, seed: int = 42
) -> Tuple[float, float, float]:
    """
    Evaluate a value-based agent in the MountainCar environment.

    Args:
        - agent: Union[ModelBasedMonteCarlo, TabularQLearning], the value-based agent to evaluate
        - max_steps: int, the maximum number of steps to take in the environment
        - n_eval_episodes: int, the number of episodes to evaluate the agent
        - seed: int, the seed for the environment

    Returns:
        - mean_reward: float, the mean reward of the agent
        - std_reward: float, the standard deviation of the reward of the agent
        - win_rate: float, the rate of successful episodes
    """
    mdp = MoutainCarMDP(discount=0.999, low=[-1.2, -0.07], high=[0.6, 0.07], num_bins=20, time_limit=1000, seed=seed)

    episode_rewards = []
    succ = 0
    for episode in tqdm(range(n_eval_episodes)):
        state = mdp.startState()
        total_rewards_ep = 0

        for step in range(max_steps):
            action = agent.getAction(state, explore=False)
            state, reward, terminated = mdp.transition(action)
            total_rewards_ep += reward

            if terminated:
                succ += 1
                break

        episode_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward, succ / n_eval_episodes


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_path", type=str, default="../results", help="Path to save results")

    args = parser.parse_args()

    policy_path = Path(args.results_path) / "reinforce/final"

    flag = policy_path.exists()

    agent_mcvi = ModelBasedMonteCarlo.from_pretrained(Path(args.results_path) / "value-iteration")
    agent_tabular = TabularQLearning.from_pretrained(Path(args.results_path) / "tabular")
    if flag:
        agent_reinforce = Policy.from_pretrained(Path(args.results_path) / "reinforce/final")

    mean_reward_mcvi, std_reward_mcvi, win_rate_mcvi = evaluate_value_agent(agent_mcvi, 200, 1000)
    mean_reward_tabular, std_reward_tabular, win_rate_tabular = evaluate_value_agent(agent_tabular, 200, 1000)
    if flag:
        mean_reward_reinforce, std_reward_reinforce = evaluate_policy_agent(agent_reinforce, 1000)

    print(f"ModelBasedMonteCarlo: {mean_reward_mcvi} +/- {std_reward_mcvi}, Win Rate: {win_rate_mcvi}")
    print(f"TabularQLearning: {mean_reward_tabular} +/- {std_reward_tabular}, Win Rate: {win_rate_tabular}")
    if flag:
        print(f"PolicyGradient: {mean_reward_reinforce} +/- {std_reward_reinforce}")

    print(f"You got a score of {win_rate_mcvi * 15 + win_rate_tabular * 15} out of 30!")


if __name__ == "__main__":
    main()
