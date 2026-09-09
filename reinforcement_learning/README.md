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

## Algorithm & Methodology

### 1. State Discretization
MountainCar-v0 provides continuous observations:
* Position: `[-1.2, 0.6]`
* Velocity: `[-0.07, 0.07]`

To use tabular Q-learning, the continuous space is mapped into a discrete grid of bins (buckets), with a default size of `(40, 40)`:
$$\text{index} = \left\lfloor \frac{\text{state} - \text{lower\_bound}}{\text{upper\_bound} - \text{lower\_bound}} \times \text{bucket\_size} \right\rfloor$$

### 2. Q-Learning Update Rule
The agent updates action values using the standard off-policy Bellman equation:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)$$

where:
* $\alpha = 0.05$ (learning rate)
* $\gamma = 0.99$ (discount factor)

where:
* $\alpha = 0.05$ (learning rate)
* $\gamma = 0.99$ (discount factor)
