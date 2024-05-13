import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import List, Tuple
import itertools


def evaluate_policy(env, q_table, episodes=1000):
    total_reward, total_length = 0, 0
    for _ in range(episodes):
        state = env.reset()[0]
        done, truncated = False, False
        episode_reward, steps = 0, 0
        while not (done or truncated):
            action = np.argmax(q_table[state])
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            state = next_state  # Ensure the state is updated
        total_reward += episode_reward
        total_length += steps
    avg_reward = total_reward / episodes
    avg_length = total_length / episodes
    return avg_reward, avg_length


def hybrid_q_learning2(env, alpha=0.1, initial_epsilon=1.0, min_epsilon=0.1, epsilon_decay=0.999, gamma=0.99, total_time_steps=10000, eval_every=1000, eval_episodes=10):
    state_space = env.observation_space.n
    action_space = env.action_space.n

    # Initialize Q-tables for A and B
    Q_A = np.random.uniform(low=0, high=0.1, size=(state_space, action_space))
    Q_B = np.copy(Q_A)

    # Initialize visitation counts
    N = np.zeros((state_space, action_space), dtype=int)

    # Arrays to store evaluation metrics
    eval_rewards, eval_lengths = [], []

    state = env.reset()[0]
    epsilon = initial_epsilon
    t = 0

    while t < total_time_steps:
        if np.random.random() < epsilon:
            action = np.random.choice(action_space)
        else:
            action = np.argmax((Q_A[state] + Q_B[state]) / 2)  # Using the average of both Q-tables

        next_state, reward, done, truncated, info = env.step(action)

        # Increment visit count for the current state-action pair
        N[state, action] += 1
        eta = 1 / N[state, action]

        # Randomly decide which Q-table to update
        if np.random.rand() < 0.5:
            best_next_action = np.argmax(Q_B[next_state])  # Best action according to Q_B
            TD_target = reward + gamma * Q_B[next_state][best_next_action]
            Q_A[state][action] += alpha * eta * (TD_target - Q_A[state][action])  # Update Q_A using Q_B
        else:
            best_next_action = np.argmax(Q_A[next_state])  # Best action according to Q_A
            TD_target = reward + gamma * Q_A[next_state][best_next_action]
            Q_B[state][action] += alpha * eta * (TD_target - Q_B[state][action])  # Update Q_B using Q_A

        state = next_state
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if done or truncated:
            state = env.reset()[0]

        if (t + 1) % eval_every == 0:
            avg_reward, avg_length = evaluate_policy(env, Q_A, eval_episodes)
            eval_rewards.append(avg_reward)
            eval_lengths.append(avg_length)
        print(t)
        t += 1

    return Q_A , eval_rewards, eval_lengths  # Return the average of both tables


def hybrid_q_learning(env, alpha=0.1, initial_epsilon=1.0, min_epsilon=0.1, epsilon_decay=0.999, gamma=0.99, total_time_steps=10000):
    state_space = env.observation_space.n
    action_space = env.action_space.n

    Q_k = np.random.uniform(low=0, high=0.1, size=(state_space, action_space))
    Q_k_minus_1 = np.copy(Q_k)
    Q_k_minus_2 = np.copy(Q_k)

    N = np.zeros((state_space, action_space), dtype=int)

    rewards, lengths = [], []
    
    state = env.reset()[0]
    epsilon = initial_epsilon

    t = 0
    k = 0

    while t <= total_time_steps:
        if np.random.random() < epsilon:
            action = np.random.choice(np.arange(Q_k.shape[1]))
        else:
            action = np.argmax(Q_k[state])
        
        next_state, reward, done, truncated, info = env.step(action)

        # Update visit count
        N[state, action] += 1
        eta = 1 / N[state, action]

        # Compute temporal differences
        best_next_action_k_minus_1 = np.argmax(Q_k_minus_1[next_state])
        best_next_action_k_minus_2 = np.argmax(Q_k_minus_2[next_state])
        T_kQ_k_minus_1 = (1 - eta) * Q_k_minus_1[state, action] + eta * (reward + gamma * Q_k_minus_1[next_state, best_next_action_k_minus_1])
        T_kQ_k_minus_2 = (1 - eta) * Q_k_minus_2[state, action] + eta * (reward + gamma * Q_k_minus_2[next_state, best_next_action_k_minus_2])

        # Update Q_k+1
        Q_k_plus_1 = (1 - alpha) * Q_k[state, action] + alpha * (k * T_kQ_k_minus_2 - (k-1) * T_kQ_k_minus_1)

        # Update Q-table references
        Q_k_minus_2 = Q_k_minus_1
        Q_k_minus_1 = Q_k
        Q_k[state, action] = Q_k_plus_1

        # Move to next state
        state = next_state

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Check if all state-action pairs have been visited
        if np.min(N) > 0:
            k += 1
            alpha = 1 / (k + 1)
            N = np.zeros_like(N)  # Reset visit counts

        t += 1

        if done:
            state = env.reset()[0]

    return Q_k


def visualize_frozen_lake(q_table, env_name="FrozenLake-v1"):
    """Visualize Frozen lake

    Args:
        q_table (_type_): learning rates
        env_name (str, optional): _description_. Defaults to "FrozenLake-v1".
    """
    env = gym.make(env_name, render_mode="human")
    state = env.reset()[0]
    env.render()
    done = False
    total_reward = 0
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        print(f"Action: {action}, State: {state}, Reward: {reward}")
        if done:
            print(f"Episode finished with a total reward of: {total_reward}")

def plot_evaluation(eval_rewards, eval_lengths):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(eval_rewards)
    plt.title("Average Cumulative Reward vs Time")
    plt.xlabel("Evaluation Period")
    plt.ylabel("Average Cumulative Reward")

    plt.subplot(1, 2, 2)
    plt.plot(eval_lengths)
    plt.title("Average Steps vs Time")
    plt.xlabel("Evaluation Period")
    plt.ylabel("Average Steps")
    plt.tight_layout()
    plt.show()


def _plot_evaluation(rewards: List[float], lengths: List[int]):
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Average Cumulative Reward vs. Evaluation Episodes")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Average Cumulative Reward")

    plt.subplot(1, 2, 2)
    plt.plot(lengths)
    plt.title("Average Steps vs. Evaluation Episodes")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Average Steps")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False)
    q_table, eval_rewards, eval_lengths = hybrid_q_learning2(env)
    plot_evaluation(eval_rewards, eval_lengths)
    print("Trained Q-Table:")
    print(q_table)