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
            if steps > 2000:
                break

        total_reward += episode_reward
        total_length += steps

    avg_reward = total_reward / episodes
    avg_length = total_length / episodes
    return avg_reward, avg_length

def moving_average(data: List[float], window_size: int) -> List[float]:
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def _plot_evaluation(rewards: List[float], lengths: List[int], epsilons: List[float], time_steps: List[int], title: str, window_size: int = 10):
    min_length = min(len(rewards), len(lengths), len(epsilons), len(time_steps))
    rewards = rewards[:min_length]
    lengths = lengths[:min_length]
    epsilons = epsilons[:min_length]
    time_steps = time_steps[:min_length]
    
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(time_steps, rewards, label='Average Reward')
    plt.title("Average Cumulative Reward (Moving Average)")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Average Cumulative Reward")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(time_steps, lengths, label='Average Steps')
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

def train_double_q_learning(
    env: gym.Env,
    alpha: float = 0.1,
    gamma: float = 0.99,
    initial_epsilon: float = 1.0,
    min_epsilon: float = 0.01,
    epsilon_decay: float = 0.999999,
    episodes: int = 50000,
    eval_every: int = 1000,
    eval_episodes: int = 100,
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

    q_table_a = initialize_random_q_table(env)
    q_table_b = initialize_random_q_table(env)

    epsilon = initial_epsilon
    rewards, lengths, epsilons, time_steps = [], [], [], []
    env.reset(seed=1234)
    first = True
    total_steps = 0
    
    for episode in range(episodes):
        state = env.reset(seed=1234)[0]
        done = False

        while not done:
            action = epsilon_greedy_policy(Q=q_table_a+q_table_b/2, state=state, epsilon=epsilon, env=env)

            next_state, reward, done, _, _ = env.step(action)
            if np.random.uniform(0,1) < 0.5:
                best_next_action = np.argmax(q_table_a[next_state, :])
                td_target = reward + gamma * q_table_b[next_state, best_next_action]
                q_table_a[state, action] += alpha * (
                    td_target - q_table_a[state, action]
                )

            else:
                best_next_action = np.argmax(q_table_b[next_state, :])
                td_target = reward + gamma * q_table_a[next_state, best_next_action]
                q_table_b[state, action] += alpha * (
                    td_target - q_table_b[state, action]
                )

            total_steps += 1
            state = next_state

            if (total_steps + 1) % eval_every == 0:
                avg_reward, avg_length = evaluate_policy(
                    env, q_table_a + q_table_b, eval_episodes
                )
                if first and avg_reward:
                    first = False
                    print(f"Episode: {episode+1}, Time Step: {total_steps + 1}, Avg. Reward: {avg_reward}, Avg. Length: {avg_length}, Epsilon: {epsilon}")
                time_steps.append(total_steps)
                rewards.append(avg_reward)
                lengths.append(avg_length)
                epsilons.append(epsilon)
                
        epsilon = max(min_epsilon, epsilon_decay * epsilon)

    print("Double Q-learning training completed.")
    return q_table_a, q_table_b, rewards, lengths, epsilons, time_steps

# Environment setup
env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False)

# Running the Double Q-learning algorithm
Q_A, Q_B, rewards, lengths, epsilons, time_steps = train_double_q_learning(env)

# Plotting the results
_plot_evaluation(rewards, lengths, epsilons, time_steps, title="Double Q-learning on FrozenLake", window_size=20)

print("Final Q_A Table:")
print(Q_A)
print("Final Q_B Table:")
print(Q_B)
