import gym
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mountain_car_qlearning import QLearningAgent

def plot_training_result(scores, avg_scores, epsilons, title):
    fig, ax1 = plt.subplots(figsize = (10, 5))

    ax1.plot(scores, color = "lightblue", alpha = 0.5, label = "Episode Reward")
    ax1.plot(avg_scores, color = "darkblue", linewidth = 2, label = "100-Ep Moving Avg")
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Reward", color = "darkblue")
    ax1.tick_params(axis = "y", labelcolor = "darkblue")
    ax1.grid(True, linestyle = "--", alpha = 0.6)

    ax2 = ax1.twinx()
    ax2.plot(epsilons, color = "red", linestyle = "-.", linewidth = 1.5, label = "Epsilon")
    ax2.set_ylabel("Epsilon (Exploration)", color = "red")
    ax2.tick_params(axis = "y", labelcolor = "red")

    plt.title(title, fontsize = 12, fontweight = "bold")
    plt.grid(True, alpha = 0.3)
    plt.show()

def run_epsilon_experiments(episodes = 100000, buckets = (40, 40), seed = 400):
    strategies = ["constant", "linear", "logarithmic", "exponential"]
    results = {}

    for strategy in strategies:
        print(f"Running strategy: {strategy}")
        agent = QLearningAgent(
            env = gym.make("MountainCar-v0"),
            buckets = buckets,
            alpha = 0.05,
            gamma = 0.99,
            epsilon_strategy = strategy,
            episodes = episodes,
            seed = seed,
        )

        scores, avgs, eps, _ = agent.train()
        results[strategy] = {"avgs": avgs, "eps": eps}
        plot_training_result(scores, avgs, eps, f"Strategy: {strategy}")

    plt.figure(figsize = (10, 5))
    for s in strategies:
        plt.plot(results[s]["avgs"], label=s)
    plt.title("Epsilon Strategies Comparison")
    plt.xlabel("Episodes")
    plt.ylabel("Moving Average Reward")
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.show()

    plt.figure(figsize = (10, 5))
    for s in strategies:
        plt.plot(results[s]["eps"], label=s)
    plt.title("Epsilon Decay Schedules")
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon")
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.show()

    return results
