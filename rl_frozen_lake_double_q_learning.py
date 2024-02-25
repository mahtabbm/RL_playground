import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import List, Tuple

def train_double_q_learning(
    env: gym.Env,
    alpha: float = 0.1,
    gamma: float = 0.99,
    initial_epsilon: float = 1.0,
    min_epsilon: float = 0.01,
    epsilon_decay: float = 0.995,
    episodes: int = 20000,
    eval_every: int = 100,
    eval_episodes: int = 10,
) -> Tuple[np.ndarray, np.ndarray, List[float], List[int]]:
    """
    Trains an agent using the Double Q-learning algorithm on a specified environment.

    Args:
        env (gym.Env): The environment to train the agent on.
        alpha (float): Learning rate.
        gamma (float): Discount factor for future rewards.
        initial_epsilon (float): Starting value for epsilon in the epsilon-greedy strategy.
        min_epsilon (float): Minimum value that epsilon can decay to over time.
        epsilon_decay (float): Rate at which epsilon decays after each episode.
        episodes (int): Total number of training episodes.
        eval_every (int): Frequency of evaluation phases during training.
        eval_episodes (int): Number of episodes to run during each evaluation phase.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[float], List[int]]: A tuple containing:
            - The final Q-table A learned by the agent.
            - The final Q-table B learned by the agent.
            - A history of average rewards obtained during evaluation periods.
            - A history of average step lengths taken during evaluation periods.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table_a = np.random.uniform(low=-0.1, high=0.1, size=(n_states, n_actions))
    q_table_b = np.random.uniform(low=-0.1, high=0.1, size=(n_states, n_actions))

    epsilon = initial_epsilon
    rewards, lengths = [], []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward, steps = 0, 0

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table_a[state, :] + q_table_b[state, :])

            next_state, reward, done, _ = env.step(action)
            if np.random.rand() < 0.5:
                best_next_action = np.argmax(q_table_a[next_state, :])
                td_target = reward + gamma * q_table_b[next_state, best_next_action]
                q_table_a[state, action] += alpha * (td_target - q_table_a[state, action])
            else:
                best_next_action = np.argmax(q_table_b[next_state, :])
                td_target = reward + gamma * q_table_a[next_state, best_next_action]
                q_table_b[state, action] += alpha * (td_target - q_table_b[state, action])

            total_reward += reward
            steps += 1

        epsilon = max(min_epsilon, epsilon_decay * epsilon)

        if (episode + 1) % eval_every == 0:
            avg_reward, avg_length = evaluate_policy(env, q_table_a + q_table_b, eval_episodes)
            rewards.append(avg_reward)
            lengths.append(avg_length)

    print("Double Q-learning training completed.")
    return q_table_a, q_table_b, rewards, lengths

# The `evaluate_policy` function remains the same as previously defined.
# The `_plot_evaluation` function remains the same as previously defined.

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True)
    q_table_a, q_table_b, rewards, lengths = train_double_q_learning(env)
    _plot_evaluation(rewards, lengths)
    print("Trained Q-Table A:")
    print(q_table_a)
    print("Trained Q-Table B:")
    print(q_table_b)
