import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import List, Tuple

def initialize_random_q_table(env):
    q_table = np.random.uniform(low=0, high=0.1, size=(env.observation_space.n, env.action_space.n))
    q_table[(env.desc == b"G").flatten()] = 0  # Ensure the goal state has Q-values of 0
    return q_table

def epsilon_greedy_policy(Q_A, Q_B, state, epsilon, env):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax((Q_A[state, :] + Q_B[state, :]) / 2)

def evaluate_policy(env, q_table, episodes=10) -> Tuple[float, float]:
    total_reward, total_length = 0, 0

    for _ in range(episodes):
        state = env.reset(seed=1234)[0]
        done = False
        episode_reward, steps = 0, 0

        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            steps += 1
            if steps > 500:
                break

        total_reward += episode_reward
        total_length += steps

    avg_reward = total_reward / episodes
    avg_length = total_length / episodes
    return avg_reward, avg_length

def moving_average(data: List[float], window_size: int) -> List[float]:
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def _plot_evaluation(rewards: List[float], lengths: List[int], epsilons: List[float], time_steps: List[int], title: str, window_size: int = 10):
    adjusted_time_steps = time_steps[len(time_steps) - len(moving_average(rewards, window_size)):]
    
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(adjusted_time_steps, moving_average(rewards, window_size), label='Average Reward')
    plt.title("Average Cumulative Reward (Moving Average)")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Average Cumulative Reward")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(adjusted_time_steps, moving_average(lengths, window_size), label='Average Steps')
    plt.title("Average Steps (Moving Average)")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Average Steps")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(time_steps, epsilons, label='Epsilon')
    plt.title("Epsilon Decay")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Epsilon Value")
    plt.legend()

    plt.tight_layout()
    plt.show()

def ultimate_sdql_training(env, epsilon=1, min_epsilon=0.1, epsilon_decay=0.999999, T=100000, gamma=0.99, eval_interval=50, eval_episodes=1):
    state_space = env.observation_space.n
    action_space = env.action_space.n

    Q_A = initialize_random_q_table(env)
    Q_B = initialize_random_q_table(env)
    Q_A_minus_1, Q_B_minus_1 = np.copy(Q_A), np.copy(Q_B)

    N_A = np.zeros((state_space, action_space))
    N_B = np.zeros((state_space, action_space))

    rewards, lengths, epsilons, time_steps = [], [], [], []

    k_A = k_B = t = 0
    alpha_A = alpha_B = 1

    state = env.reset(seed=1234)[0]

    lake_map = env.desc
    frozen_lake_binary = np.array((lake_map != b'H') & (lake_map != b'G')).astype(int).flatten()

    first = True
    while t <= T:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            combined_Q = (Q_A[state, :] + Q_B[state, :]) / 2
            action = np.argmax(combined_Q)

        next_state, reward, done, _, _ = env.step(action)

        if np.random.uniform(0, 1) < 0.5:  # Update Q_A
            eta = 1 / (N_A[state, action] + 1)
            a_star = np.argmax(Q_A_minus_1[next_state, :])
            a_plus = np.argmax(Q_A[next_state, :])

            T_kQ_A_minus_1 = (1 - eta) * Q_A_minus_1[state, action] + eta * (reward + gamma * Q_B_minus_1[next_state, a_star])
            T_kQ_A = (1 - eta) * Q_A[state, action] + eta * (reward + gamma * Q_B[next_state, a_plus])

            Q_A_minus_1[state, action] = Q_A[state, action]
            Q_A[state, action] = (1 - alpha_A) * Q_A[state, action] + alpha_A * (k_A * T_kQ_A - (k_A - 1) * T_kQ_A_minus_1)
            N_A[state, action] += 1
        else:  # Update Q_B
            eta = 1 / (N_B[state, action] + 1)
            a_star = np.argmax(Q_B_minus_1[next_state, :])
            a_plus = np.argmax(Q_B[next_state, :])
            T_kQ_B_minus_1 = (1 - eta) * Q_B_minus_1[state, action] + eta * (reward + gamma * Q_A_minus_1[next_state, a_star])
            T_kQ_B = (1 - eta) * Q_B[state, action] + eta * (reward + gamma * Q_A[next_state, a_plus])

            Q_B_minus_1[state, action] = Q_B[state, action]
            Q_B[state, action] = (1 - alpha_B) * Q_B[state, action] + alpha_B * (k_B * T_kQ_B - (k_B - 1) * T_kQ_B_minus_1)
            N_B[state, action] += 1

        state = next_state

        # Update counters and learning rates if needed
        if np.min(N_A[frozen_lake_binary == 1]) > 0:
            k_A += 1
            alpha_A = 1 / (k_A + 1)
            N_A.fill(0)  # Reset visit counts for A
        
        if np.min(N_B[frozen_lake_binary == 1]) > 0:
            k_B += 1
            alpha_B = 1 / (k_B + 1)
            N_B.fill(0)  # Reset visit counts for B

        epsilon = max(min_epsilon, epsilon * epsilon_decay)  # Reduce epsilon
        t += 1
        if done:
            state = env.reset(seed=1234)[0]

        if (t + 1) % eval_interval == 0:
            avg_reward, avg_length = evaluate_policy(env, Q_A + Q_B, eval_episodes)
            if first and avg_reward:
                first = False
                print(f"Time Step = {t + 1}: Avg Reward = {avg_reward}, Avg Length = {avg_length}, epsilon = {epsilon}")
            time_steps.append(t + 1)
            rewards.append(avg_reward)
            lengths.append(avg_length)
            epsilons.append(epsilon)

    return Q_A + Q_B, rewards, lengths, epsilons, time_steps

# Environment setup
env = gym.make('FrozenLake-v1', is_slippery=False)

# Running the Ultimate SDQL algorithm
q_table, rewards, lengths, epsilons, time_steps = ultimate_sdql_training(env, T=200000, eval_interval=1000, eval_episodes=10)

# Plotting the results
_plot_evaluation(rewards, lengths, epsilons, time_steps, title="Ultimate SDQL on FrozenLake", window_size=5)

print("Final Q-Table:")
print(q_table)
