# Mountain Car Q-Learning

A tabular Q-learning implementation to solve the classic MountainCar-v0 control problem by discretizing its continuous state space.

## Overview

The Mountain Car problem consists of an underpowered car placed in a valley between two hills. The goal is to drive up the steep hill on the right to reach the flag. Because gravity is stronger than the car's engine, the agent must build momentum by moving back and forth between the two hills.

This project implements:
* State discretization to convert continuous position and velocity into discrete grid bins.
* Standard Q-learning algorithm with a Bellman update rule.
* Multiple exploration strategies to compare learning stability and convergence speed.

## Repository Structure

* `mountain_car_qlearning.py`: Contains the main `QLearningAgent` class, discretization logic, Q-table update rule, and the `watch_agent` helper to render the trained car.
* `experiments.py`: Runs baseline training, generates the 3D value function surface, and compares different epsilon decay strategies and bucket grid sizes.
