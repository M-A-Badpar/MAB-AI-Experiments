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
