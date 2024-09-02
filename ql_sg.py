import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import List, Tuple
import gym_simplegrid

def initialize_random_q_table(env, goal_state):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))
    q_table[goal_state, :] = 0  # Ensure the goal state has a Q-value of 0
    return q_table

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

def q_learning(
    env: gym.Env,
    alpha: float = 0.1,
    gamma: float = 0.99,
    initial_epsilon: float = 1.0,
    min_epsilon: float = 0.01,
    epsilon_decay: float = 0.999,
    episodes: int = 20000,
    eval_every: int = 100,
    eval_episodes: int = 20,
    goal_state = 15,
) -> Tuple[np.ndarray, List[float], List[float], List[float]]:
    """Trains an agent using the Q-learning algorithm on a specified environment."""
    q_table = initialize_random_q_table(env, goal_state)
    options = {
        'start_loc': 0,
        'goal_loc': goal_state
    }

    env.reset(seed=1234, options=options)
    epsilon = initial_epsilon
    rewards, lengths, epsilons, total_steps = [], [], [], []
    first = True
    steps = 0

    for episode in range(episodes):
        state = env.reset(seed=1234, options=options)[0]
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.uniform(0, 1) <= epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state, :])  # Exploit learned values

            next_state, reward, done, truncated, info = env.step(action)

            # Q-Learning update rule
            q_table[state, action] += alpha * (
                reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
            )
            state = next_state
            steps += 1

            if steps % eval_every == 0:
                avg_reward, avg_length = evaluate_policy(env, q_table, eval_episodes)
                if first and avg_reward == 1:
                    first = False
                    print(f"Total Time Steps: {steps}, Episode: {episode + 1}, Avg. Reward: {avg_reward}, Avg. Length: {avg_length}, Epsilon: {epsilon}")
                total_steps.append(steps)
                rewards.append(avg_reward)
                lengths.append(avg_length)
                epsilons.append(epsilon)

        # Epsilon decay
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    print("Training completed.")
    return q_table, rewards, lengths, epsilons, total_steps

# Define obstacle map for SimpleGrid
obstacle_map = [
    "0000",
    "0101",
    "0001",
    "1000",
]

# Environment setup for SimpleGrid 4x4
env = gym.make('SimpleGrid-4x4-v0', render_mode='rgb_array', obstacle_map=obstacle_map)

# Running the Q-learning algorithm
q_table, rewards, lengths, epsilons, time_steps = q_learning(env, episodes=100, eval_episodes=10, eval_every=100)

# Plotting the results
_plot_evaluation(rewards, lengths, epsilons, time_steps, title="Q-learning on SimpleGrid 4x4", window_size=5)

print("Final Q-Table:")
print(q_table)
