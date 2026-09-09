import math
import gym
import numpy as np
from collections import deque

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

    def reset_env(self, episode=0):
        reset_result = self.env.reset(seed=self.seed + episode)
        if isinstance(reset_result, tuple):
            obs = reset_result[0]
        else:
            obs = reset_result
        return obs

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
            return max(0.01, 1.0 / (1.0 + 0.1 * math.log(1 + episode)))
        return 0.1

    def train(self):
        scores_window = deque(maxlen=100)

        scores_array = []
        avg_scores_array = []
        epsilons_array = []
        shaped_scores_array = []

        for episode in range(self.episodes):
            obs = self.reset_env(episode)
            state = self.discretize_state(obs)
            eps = self.get_epsilon(episode)

            done = False
            current_score = 0

            while not done:
                action = self.choose_action(state, eps)

                step_result = self.env.step(action)
                if len(step_result) == 4:
                    next_obs, reward, done, _ = step_result
                else:
                    next_obs, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated

                next_state = self.discretize_state(next_obs)

                best_next = np.argmax(self.Q_table[next_state])
                self.Q_table[state][action] += self.alpha * (
                    reward
                    + self.gamma * self.Q_table[next_state][best_next]
                    - self.Q_table[state][action]
                )

                state = next_state
                current_score += reward

            scores_array.append(current_score)
            scores_window.append(current_score)
            mean_score = np.mean(scores_window)
            avg_scores_array.append(mean_score)
            epsilons_array.append(eps)
            shaped_scores_array.append(current_score)

            if (episode + 1) % 2000 == 0:
                print(f"Ep {episode+1} -> Avg: {mean_score:.1f}, Eps: {eps:.3f}")

            if mean_score >= -110.0 and len(scores_window) >= 100:
                print(f"Goal reached at episode {episode+1}! Mean score: {mean_score:.2f}")
                break

        return scores_array, avg_scores_array, epsilons_array, shaped_scores_array

    def evaluate(self, episodes=100):
        eval_scores = []

        for episode in range(episodes):
            obs = self.reset_env(episode)
            state = self.discretize_state(obs)
            done = False
            score = 0

            while not done:
                action = int(np.argmax(self.Q_table[state]))
                step_result = self.env.step(action)
                if len(step_result) == 4:
                    next_obs, reward, done, _ = step_result
                else:
                    next_obs, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                state = self.discretize_state(next_obs)
                score += reward
            eval_scores.append(score)
            
        return eval_scores

    def watch_agent(agent, episodes=5):
    env = gym.make("MountainCar-v0", render_mode="human")

    for episode in range(episodes):
        reset_result = env.reset(seed=agent.seed + 200000 + episode)

        if isinstance(reset_result, tuple):
            obs = reset_result[0]
        else:
            obs = reset_result

        state = agent.discretize_state(obs)
        done = False
        total_reward = 0

        while not done:
            action = int(np.argmax(agent.Q_table[state]))

            step_result = env.step(action)

            if len(step_result) == 4:
                next_obs, reward, done, _ = step_result
            else:
                next_obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated

            state = agent.discretize_state(next_obs)
            total_reward += reward

        print(f"Episode {episode + 1} reward: {total_reward}")

    env.close()
