import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import List, Tuple

def initialize_random_q_table(env):
    q_table = np.random.uniform(low=0, high=0.1, size=(env.observation_space.n, env.action_space.n))
    q_table[(env.desc == b"G").flatten()] = 0  # Assuming 'G' is the goal/terminal state
    return q_table

def epsilon_greedy_policy(Q, state, epsilon, env):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state, :])

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
            if steps > 1000:
                break

        total_reward += episode_reward
        total_length += steps

    avg_reward = total_reward / episodes
    avg_length = total_length / episodes
    return avg_reward, avg_length

def moving_average(data: List[float], window_size: int) -> List[float]:
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def _plot_evaluation(rewards: List[float], lengths: List[int], epsilons: List[float], time_steps: List[int], title: str, window_size: int = 10):
    # Adjust the time_steps to match the length of the moving average
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

def asynchronous_speedy_q_learning(
        env, epsilon=1.0, 
        gamma=0.99, 
        total_time_steps=10000,
        eval_every: int = 1000,
        eval_episodes: int = 100,         
        epsilon_decay=0.9999999,
        min_epsilon = 0.01
):
    state_space = env.observation_space.n
    action_space = env.action_space.n
    alpha = 1
    # Initialize Q-tables and state visit counts
    Q_k = initialize_random_q_table(env)
    Q_k_minus_1 = np.copy(Q_k)
    k = 0  # Iteration counter
    N = np.zeros((state_space, action_space), dtype=int)
    state = env.reset(seed=1234)[0]
    t = 0
    rewards, lengths, epsilons, time_steps = [], [], [], []
    first = True

    lake_map = env.desc
    frozen_lake_binary = np.array((lake_map != b'H') & (lake_map != b'G')).astype(int).flatten()

    while t <= total_time_steps:
        action = epsilon_greedy_policy(Q_k, state, epsilon, env)
        next_state, reward, done, _, _ = env.step(action)
        # Update visit count

        eta = 1 / (N[state][action] + 1)

        # Compute temporal differences
        best_next_action_k_minus_1 = np.argmax(Q_k_minus_1[next_state])
        best_next_action_k = np.argmax(Q_k[next_state])
        T_kQ_k_minus_1 = (1 - eta) * Q_k_minus_1[state, action] + eta * (reward + gamma * Q_k_minus_1[next_state, best_next_action_k_minus_1])
        T_kQ_k = (1 - eta) * Q_k[state, action] + eta * (reward + gamma * Q_k[next_state, best_next_action_k])

        # Update Q_k+1
        Q_k_plus_1 = (1 - alpha) * Q_k[state, action] + alpha * (k * T_kQ_k - (k-1) * T_kQ_k_minus_1)
        N[state][action] += 1

        # Update Q-table references
        Q_k_minus_1 = np.copy(Q_k)
        Q_k[state, action] = np.copy(Q_k_plus_1)
        # Move to next state
        state = next_state

        # Check if all state-action pairs have been visited
        if np.min(N[frozen_lake_binary == 1]) > 0:
            k += 1
            alpha = 1 / (k + 1)
            N = np.zeros_like(N)  # Reset visit counts

        t += 1

        epsilon = max(min_epsilon, epsilon_decay * epsilon)

        if done:
            state = env.reset(seed=1234)[0]
            
        if (t + 1) % eval_every == 0:
            avg_reward, avg_length = evaluate_policy(env, Q_k, eval_episodes)
            if first and avg_reward:
                first = False
                print(f"Time Step: {t + 1}, Avg. Reward: {avg_reward}, Avg. Length: {avg_length}, Epsilon: {epsilon}")   
            time_steps.append(t+1)
            rewards.append(avg_reward)
            lengths.append(avg_length)
            epsilons.append(epsilon)

    return Q_k, rewards, lengths, epsilons, time_steps

# Environment setup
env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False)

# Running the Asynchronous Speedy Q-learning algorithm
Q_k, rewards, lengths, epsilons, time_steps = asynchronous_speedy_q_learning(env, total_time_steps=1000000)

# Plotting the results
_plot_evaluation(rewards, lengths, epsilons, time_steps, title="Asynchronous Speedy Q-learning on FrozenLake", window_size=20)

print("Final Q-Table:")
print(Q_k)
