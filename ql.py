import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import List, Tuple

# Reusing the existing functions for evaluation, moving average, and plotting

def evaluate_policy(env, q_table, episodes=10) -> Tuple[float, float]:
    total_reward, total_length = 0, 0

    for _ in range(episodes):
        state = env.reset(seed=1234)[0]
        done = False
        episode_reward, steps = 0, 0

        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, truncated, info = env.step(action)
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
    """Compute the moving average of a list of numbers."""
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def _plot_evaluation(rewards: List[float], lengths: List[int], epsilons: List[float], time_steps: List[int], title: str, window_size: int = 10):
    """Plot the exponential moving average of rewards, lengths, and epsilon values."""
    
    # Ensure that all lists are the same length
    min_length = min(len(rewards), len(lengths), len(epsilons), len(time_steps))
    rewards = rewards[:min_length]
    lengths = lengths[:min_length]
    epsilons = epsilons[:min_length]
    time_steps = time_steps[:min_length]
    
    plt.figure(figsize=(18, 5))

    # Plot for average cumulative rewards
    plt.subplot(1, 3, 1)
    plt.plot(time_steps, rewards, label='Average Reward')
    plt.title("Average Cumulative Reward (Moving Average)")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Average Cumulative Reward")
    plt.legend()

    # Plot for average steps
    plt.subplot(1, 3, 2)
    plt.plot(time_steps, lengths, label='Average Steps')
    plt.title("Average Steps (Moving Average)")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Average Steps")
    plt.legend()

    # Plot for epsilon changes
    plt.subplot(1, 3, 3)
    plt.plot(time_steps, epsilons, label='Epsilon')
    plt.title("Epsilon Decay")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Epsilon Value")
    plt.legend()

    plt.tight_layout()
    plt.show()

def average_results(results: List[Tuple[List[float], List[float], List[float], List[int]]], num_evals: int) -> Tuple[List[float], List[float], List[float]]:
    # Determine the minimum length of all result lists to align them
    min_length = min(len(rewards) for rewards, _, _, _ in results)

    avg_rewards = np.zeros(min_length)
    avg_lengths = np.zeros(min_length)
    avg_epsilons = np.zeros(min_length)

    for rewards, lengths, epsilons, _ in results:  # Now unpacking four elements
        avg_rewards += np.array(rewards[:min_length])
        avg_lengths += np.array(lengths[:min_length])
        avg_epsilons += np.array(epsilons[:min_length])

    avg_rewards /= len(results)
    avg_lengths /= len(results)
    avg_epsilons /= len(results)

    return avg_rewards.tolist(), avg_lengths.tolist(), avg_epsilons.tolist()

def initialize_random_q_table(env):
    q_table = np.random.uniform(low=0, high=0.1, size=(env.observation_space.n, env.action_space.n))
    q_table[(env.desc == b"G").flatten()] = 0  # Assuming 'G' is the goal/terminal state
    return q_table

def epsilon_greedy_policy(Q, state, epsilon, env):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state, :])

def train_q_learning(
    env: gym.Env,
    alpha: float = 0.1,
    gamma: float = 0.99,
    initial_epsilon: float = 1.0,
    min_epsilon: float = 0.01,
    epsilon_decay: float = 0.9999,
    episodes: int = 10000,
    eval_every: int = 1000,
    eval_episodes: int = 50,
) -> Tuple:
    """Trains an agent using the Q-learning algorithm on a specified environment."""

    q_table = initialize_random_q_table(env)
    env.reset(seed=1234)
    epsilon = initial_epsilon
    rewards, lengths, epsilons, time_steps = [], [], [], []
    first = True
    total_steps = 0

    for episode in range(episodes):
        state = env.reset(seed=1234)[0]
        done = False

        while not done:
            # Epsilon-greedy action selection
            action = epsilon_greedy_policy(q_table, state, epsilon, env)

            next_state, reward, done, _, _ = env.step(action)

            # Q-Learning update rule
            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
            )
            state = next_state

            total_steps += 1

            # Evaluation
            if (total_steps + 1) % eval_every == 0:
                avg_reward, avg_length = evaluate_policy(env, q_table, eval_episodes)
                if first and avg_reward:
                    first = False
                    print(f"Episode: {episode+1}, Time Step: {total_steps + 1}, Avg. Reward: {avg_reward}, Avg. Length: {avg_length}, Epsilon: {epsilon}")
                time_steps.append(total_steps+1)
                rewards.append(avg_reward)
                lengths.append(avg_length)
                epsilons.append(epsilon)

        # Epsilon decay
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    print("Training completed.")
    return q_table, rewards, lengths, epsilons, time_steps

# Environment setup
env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False)

num_runs = 3
all_results = []

for run in range(num_runs):
    print(f"Training run {run + 1}")
    q_table, rewards, lengths, epsilons, time_steps = train_q_learning(env)
    all_results.append((rewards, lengths, epsilons, time_steps))

avg_rewards, avg_lengths, avg_epsilons = average_results(all_results, num_evals=len(all_results[0][0]))

# Plotting the averaged metrics
_plot_evaluation(avg_rewards, avg_lengths, avg_epsilons, time_steps, title="Q-learning FL4x4 (Averaged over 3 runs)", window_size=5)
print(q_table)
