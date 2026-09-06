import math
import gym
import numpy as np

class QLearningAgent:
  def __init__(
      self,
      env,
      buckets=(40, 40),
      alpha=0.05,
      gamma=0.99,
      epsilon_strategy="exponential",
      episodes=100000,
      seed=400,
  ):
    self.env = env
    self.buckets = buckets
    self.alpha = alpha
    self.gamma = gamma
    self.episodes = episodes
    self.epsilon_strategy = epsilon_strategy
    self.seed = seed
    self.rng = np.random.default_rng(seed)
    self.env.action_space.seed(seed)
    self.lower_bounds = self.env.observation_space.low
    self.upper_bounds = self.env.observation_space.high
    self.Q_table = np.zeros(self.buckets + (self.env.action_space.n,))

  def discretize_state(self, obs):
    scaled = (obs - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)
    
    scaled = np.clip(scaled, 0.0, 1.0)

    idx_position = int(scaled[0] * (self.buckets[0] - 1))
    idx_velocity = int(scaled[1] * (self.buckets[1] - 1))
    
    return (idx_position, idx_velocity)

def choose_action(self, state, epsilon):
    if self.rng.random() < epsilon:
        return self.env.action_space.sample()
    else:
        return int(np.argmax(self.Q_table[state]))

def get_epsilon(self, episode):
    if self.epsilon_strategy == "constant":
        return 0.1
    elif self.epsilon_strategy == "linear":
        return max(0.01, 1.0 - episode / self.episodes)
    elif self.epsilon_strategy == "exponential":
        return max(0.01, 1.0 * math.exp(-0.0001 * episode))
    elif self.epsilon_strategy == "logarithmic":
        return max(0.01, 1.0 / (1.0 + 0.1*math.log(1 + episode)))
    return 0.1
