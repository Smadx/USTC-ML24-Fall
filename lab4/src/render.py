from utils import RandomAgent, simulate
from mdp import MoutainCarMDP

from pathlib import Path

import time
import torch
import numpy as np
import gymnasium as gym
import argparse

from submission import ModelBasedMonteCarlo, TabularQLearning, Policy

if __name__ == "__main__":
    """
    The main function called when render.py is run
    from the command line:

    > python render.py

    See the usage string for more details.

    > python render.py --help
    """
    # play.play(gym.make("MountainCar-v0", render_mode="human"), zoom=3)
    # TODO: Implement interactive mode for human play
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["naive-mountaincar", "naive-cartpole", "value-iteration", "tabular", "reinforce"],
        help='naive-mountaincar ("naive-mountaincar"), naive-cartpole ("naive-cartpole"), model-based value iteration ("value-iteration"), tabular Q-learning ("tabular"), policy gradient ("reinforce")',
    )
    parser.add_argument("--results_path", type=str, default="../results", help="Path to save results")
    args = parser.parse_args()

    model_path = Path(args.results_path) / args.agent

    # Naive Agent
    if args.agent == "naive-mountaincar":
        print("************************************************")
        print("Naive agent performing mountain car task!")
        print("************************************************")
        mdp = MoutainCarMDP(discount=0.999, time_limit=1000)
        mdp.env = gym.make("MountainCar-v0", render_mode="human")
        rl = RandomAgent(mdp.actions)
        simulate(mdp, rl, train=False, numTrials=1, verbose=False, demo=True)
        mdp.env.close()

    # Naive Agent
    elif args.agent == "naive-cartpole":
        print("************************************************")
        print("Naive agent performing cartpole task!")
        print("************************************************")
        env = gym.make("CartPole-v1", render_mode="human")
        state, _ = env.reset()
        env.render()
        while True:
            time.sleep(0.1)
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.close()

    # Agent Trained w/ Model-Based Value Iteration
    elif args.agent == "value-iteration":
        print("********************************************************")
        print("Agent trained with model-based value iteration performing mountain car task!")
        print("********************************************************")
        mdp = MoutainCarMDP(
            discount=0.999,
            low=[-1.2, -0.07],
            high=[0.6, 0.07],
            num_bins=20,
            time_limit=1000,
        )
        mdp.env = gym.make("MountainCar-v0", render_mode="human")
        rl = ModelBasedMonteCarlo.from_pretrained(model_path)
        simulate(mdp, rl, train=False, numTrials=1, verbose=False, demo=True)
        mdp.env.close()

    # Agent Trained w/ Tabular Q-Learning
    elif args.agent == "tabular":
        print("********************************************************")
        print("Agent trained with Tabular Q-Learning performing mountain car task!")
        print("********************************************************")
        mdp = MoutainCarMDP(
            discount=0.999,
            low=[-1.2, -0.07],
            high=[0.6, 0.07],
            num_bins=20,
            time_limit=1000,
        )
        mdp.env = gym.make("MountainCar-v0", render_mode="human")
        rl = TabularQLearning.from_pretrained(model_path)
        simulate(mdp, rl, train=False, numTrials=1, verbose=False, demo=True)
        mdp.env.close()

    # Agent Trained w/ Policy Gradient
    elif args.agent == "reinforce":
        print("********************************************************")
        print("Agent trained with Policy Gradient performing mountain car task!")
        print("********************************************************")
        agent_path = model_path / "final"
        rl = Policy.from_pretrained(agent_path)
        env = gym.make("CartPole-v1", render_mode="human")

        state, _ = env.reset()
        env.render()
        while True:
            time.sleep(0.05)
            state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
            action, _ = rl.getAction(state)
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.close()
    else:
        raise ValueError("Invalid agent type")
