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

def run_bucket_experiments(episodes = 100000, seed = 400):
    grids = [(5, 5), (100, 100)]
    results = {}

    print("\nRunning Discretization Bucket Experiments")

    for grid in grids:
        print(f"Running grid size: {grid}")
        agent = QLearningAgent(
            env = gym.make("MountainCar-v0"),
            buckets = grid,
            alpha = 0.05,
            gamma = 0.99,
            epsilon_strategy = "exponential",
            episodes = episodes,
            seed = seed,
        )

        _, avgs, _, _ = agent.train()
        results[str(grid)] = avgs

    plt.figure(figsize = (10, 5))
    for grid in grids:
        plt.plot(results[str(grid)], label = f"Grid {grid}")

    plt.title("Discretization Bucket Sizes Comparison", fontsize = 12, fontweight = "bold")
    plt.xlabel("Episodes")
    plt.ylabel("Moving Average Reward")
    plt.legend()
    plt.grid(True, alpha = 0.3)
    plt.show()

    return results

def plot_value_function(agent):
    value_table = np.max(agent.Q_table, axis = 2)

    unvisited = np.all(agent.Q_table == 0, axis = 2)
    value_table_masked = np.array(value_table, dtype = float)
    value_table_masked[unvisited] = np.nan

    pos_min, pos_max = agent.lower_bounds[0], agent.upper_bounds[0]
    vel_min, vel_max = agent.lower_bounds[1], agent.upper_bounds[1]

    positions = np.linspace(pos_min, pos_max, agent.buckets[0])
    velocities = np.linspace(vel_min, vel_max, agent.buckets[1])

    X, Y = np.meshgrid(positions, velocities)
    Z = value_table_masked.T

    fig = plt.figure(figsize = (10, 5))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap = "viridis", edgecolor = "none", alpha = 0.9)

    ax.set_title(r"3D State-Value Function $V(s) = \max_a Q(s, a)$", fontsize = 12, fontweight = "bold")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("Value V(s)")

    fig.colorbar(surf, ax = ax, shrink = 0.5, aspect = 10)
    plt.tight_layout()
    plt.show()
