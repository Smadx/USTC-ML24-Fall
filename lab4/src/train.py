import yaml
import argparse
from mdp import MoutainCarMDP
import numpy as np
from dataclasses import asdict
from matplotlib import pyplot as plt
from typing import Union
from pathlib import Path

from utils import TrainConfig, Metrics, simulate, handle_results_path, init_config_from_args

from grade import evaluate_policy_agent

from submission import ModelBasedMonteCarlo, TabularQLearning, Policy, reinforce


def movingAverage(x, window):
    cumSum = np.cumsum(x)
    ma = (cumSum[window:] - cumSum[:-window]) / window
    return ma


def plotRewards(trainRewards: list, evalRewards: list, savePath: Union[str, Path] = None, show: bool = True):
    """
    Plot the rewards from training and evaluation episodes

    Args:
        - trainRewards: list, the rewards from training episodes
        - evalRewards: list, the rewards from evaluation episodes
        - savePath: str, the path to save the plot
        - show: bool, whether to display the plot
    """
    plt.figure(figsize=(10, 5))
    window = 30
    trainMA = movingAverage(trainRewards, window)
    evalMA = movingAverage(evalRewards, window)
    tLen = len(trainRewards)
    eLen = len(evalRewards)
    plt.scatter(range(tLen), trainRewards, alpha=0.5, c="tab:blue", linewidth=0, s=5)
    plt.plot(range(int(window / 2), tLen - int(window / 2)), trainMA, lw=2, c="b")
    plt.scatter(range(tLen, tLen + eLen), evalRewards, alpha=0.5, c="tab:green", linewidth=0, s=5)
    plt.plot(range(tLen + int(window / 2), tLen + eLen - int(window / 2)), evalMA, lw=2, c="darkgreen")
    plt.legend(["train rewards", "train moving average", "eval rewards", "eval moving average"])
    plt.xlabel("Episode")
    plt.ylabel("Discounted Reward in Episode")

    if savePath is not None:
        plt.savefig(savePath)
    if show:
        plt.show()


if __name__ == "__main__":
    """
    The main function called when train.py is run
    from the command line:

    > python train.py

    See the usage string for more details.

    > python train.py --help
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["value-iteration", "tabular", "reinforce"],
        help='model-based value iteration ("value-iteration"), tabular Q-learning ("tabular"), policy gradient ("reinforce")',
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default=None,
        help="Path to save results",
    )
    parser.add_argument("--mcvi_exprob", type=float, default=0.5, help="ExplorationProb for mcvi training.")
    parser.add_argument("--mcvi_episodes", type=int, default=1000, help="The number of episodes for mcvi training.")
    parser.add_argument(
        "--tabular_exprob", type=float, default=0.15, help="ExplorationProb for TabularQLearning training."
    )
    parser.add_argument(
        "--tabular_episodes", type=int, default=1000, help="The number of episodes for TabularQLearning Training."
    )

    # If you don't want to try Policy Gradient, the above arguments are enough.
    # [Optional] Policy Gradient Training
    parser.add_argument("--is_resume", action="store_true", help="Resume training from a pretrained checkpoint.")
    parser.add_argument("--pretrain_path", type=str, default=None, help="Path to pretrained checkpoint.")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of trajectories uesd to estimate policy gradient."
    )
    parser.add_argument("--num_updates", type=int, default=1000, help="Number of policy updates to perform.")
    parser.add_argument("--max_t", type=int, default=1000, help="Maximum time steps in an episode.")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor for rewards.")
    parser.add_argument("--lr", type=float, default=2e-2, help="Learning rate for policy gradient updates.")
    parser.add_argument("--save_every", type=int, default=100, help="The number of steps to save a checkpoint.")
    parser.add_argument("--window_size", type=int, default=100, help="The number of steps to log average reward.")

    # If you don't want to track your experiment in wandb, the above arguments are enough.
    # [Optional] Tracking your experiment in wandb
    parser.add_argument("--track", action="store_true", help="Use wandb to track experiments.")
    parser.add_argument("--group", type=str, default=None, help="Experiment group.")
    parser.add_argument("--project", type=str, default="CartPole", help="Project name.")
    parser.add_argument("--run_id", type=str, default=None, help="Name of a run.")
    parser.add_argument("--entity", type=str, default=None, help="Name of a Team or Person, None for your username.")
    parser.add_argument("--wandb_dir", type=str, default=None, help="Local dir for wandb.")

    args = parser.parse_args()

    cfg = init_config_from_args(TrainConfig, args)

    results_path = handle_results_path(cfg.results_path) / Path(cfg.agent)
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / "config.yaml", "w") as f:
        yaml.dump(asdict(cfg), f)

    save_path = results_path
    save_path.mkdir(parents=True, exist_ok=True)

    if cfg.agent == "value-iteration":
        print("************************************************")
        print("Training agent with model-based value iteration to perform mountain car task!")
        print("************************************************")
        for i in range(1, 4):
            print("********************************************************")
            print(f"Trial {i} out of 3")
            print("********************************************************")
            mdp = MoutainCarMDP(
                discount=0.999,
                low=[-1.2, -0.07],
                high=[0.6, 0.07],
                num_bins=20,
                time_limit=1000,
            )
            rl = ModelBasedMonteCarlo(mdp.actions, mdp.discount, calcValIterEvery=1e5, explorationProb=cfg.mcvi_exprob)
            trainRewards = simulate(mdp, rl, train=True, numTrials=cfg.mcvi_episodes, verbose=True)
            print(
                f"Training complete! Running evaluation, writing weights to {save_path}/mcvi.safetensors and generating reward plot..."
            )
            evalRewards = simulate(mdp, rl, train=False, numTrials=500)

            rl.save_pretrained(save_path)

            plotRewards(trainRewards, evalRewards, save_path / f"mcvi_{i}.png")

    # Trained Discrete Agent
    elif cfg.agent == "tabular":
        print("********************************************************")
        print("Training agent with Tabular Q-Learning to perform mountain car task!")
        print("********************************************************")
        for i in range(1, 4):
            print("********************************************************")
            print(f"Trial {i} out of 3")
            print("********************************************************")
            mdp = MoutainCarMDP(
                discount=0.999,
                low=[-1.2, -0.07],
                high=[0.6, 0.07],
                num_bins=20,
                time_limit=1000,
            )
            rl = TabularQLearning(mdp.actions, mdp.discount, explorationProb=cfg.tabular_exprob)
            trainRewards = simulate(mdp, rl, train=True, numTrials=cfg.tabular_episodes, verbose=True)
            print(
                f"Training complete! Running evaluation, writing weights to {save_path}/tabular.safetensors and generating reward plot..."
            )
            evalRewards = simulate(mdp, rl, train=False, numTrials=500)

            rl.save_pretrained(save_path)

            plotRewards(trainRewards, evalRewards, save_path / f"tabular_{i}.png")

    # Training Policy
    elif cfg.agent == "reinforce":
        print("********************************************************")
        print("Training agent with Policy Gradient to perform mountain car task!")
        print("********************************************************")

        if not cfg.is_resume:
            policy = Policy()
            stage = "new-stage"
        else:
            policy = Policy.from_pretrained(cfg.pretrain_path)
            reward, std = evaluate_policy_agent(policy, cfg.max_t, 100)
            print(f"Pre-trained policy mean reward: {reward}, std: {std}")
            stage = "from-pretrained"

        if cfg.track:
            metrics = Metrics(
                stage=stage,
                run_id=cfg.run_id,
                run_dir=cfg.wandb_dir,
                hparams=asdict(cfg),
                wandb_project=cfg.project,
                wandb_entity=cfg.entity,
                window_size=cfg.window_size,
            )

        policy = reinforce(
            policy=policy,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            num_updates=cfg.num_updates,
            max_t=cfg.max_t,
            gamma=cfg.gamma,
            checkpoint_path=save_path,
            save_every=cfg.save_every,
            window_size=cfg.window_size,
            metrics=metrics if cfg.track else None,
        )

        reward, std = evaluate_policy_agent(policy, 100)
        print(f"Post-trained policy mean reward: {reward}, std: {std}")
