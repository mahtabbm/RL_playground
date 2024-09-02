import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from typing import List, Tuple

# Define a function to run a given algorithm
def run_algorithm(env, algorithm_func, **kwargs):
    q_table, rewards, lengths, epsilons, time_steps = algorithm_func(env, **kwargs)
    return rewards, lengths, epsilons, time_steps

# Plotting function to compare algorithms on the same plot
def compare_algorithms(results, titles, window_size=100):
    plt.figure(figsize=(18, 6))

    # Plot Average Cumulative Rewards
    plt.subplot(1, 3, 1)
    for i, (rewards, _, _, time_steps) in enumerate(results):
        rewards_ma = moving_average(rewards, window_size)
        adjusted_time_steps = time_steps[-len(rewards_ma):]
        plt.plot(adjusted_time_steps, rewards_ma, label=f'{titles[i]} Reward')
    plt.title("Average Cumulative Reward")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Average Cumulative Reward")
    plt.legend()

    # Plot Average Steps
    plt.subplot(1, 3, 2)
    for i, (_, lengths, _, time_steps) in enumerate(results):
        lengths_ma = moving_average(lengths, window_size)
        adjusted_time_steps = time_steps[-len(lengths_ma):]
        plt.plot(adjusted_time_steps, lengths_ma, label=f'{titles[i]} Steps')
    plt.title("Average Steps")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Average Steps")
    plt.legend()

    # Plot Epsilon Decay
    plt.subplot(1, 3, 3)
    for i, (_, _, epsilons, time_steps) in enumerate(results):
        plt.plot(time_steps, epsilons, label=f'{titles[i]} Epsilon')
    plt.title("Epsilon Decay")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Epsilon Value")
    plt.legend()

    plt.tight_layout()
    plt.show()

def moving_average(data: List[float], window_size: int) -> List[float]:
    if len(data) < window_size:
        return np.array(data)  # Return the data as is if it's shorter than the window size
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def moving_average(data: List[float], window_size: int) -> List[float]:
    if len(data) < window_size:
        return np.array(data)  # Return the data as is if it's shorter than the window size
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def initialize_random_q_table(env):
    q_table = np.random.uniform(low=0, high=0.1, size=(env.observation_space.n, env.action_space.n))
    q_table[(env.desc == b"G").flatten()] = 0  # Ensure the goal state has Q-values of 0
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
            if steps > 500:
                break

        total_reward += episode_reward
        total_length += steps

    avg_reward = total_reward / episodes
    avg_length = total_length / episodes
    return avg_reward, avg_length


def train_q_learning(
    env: gym.Env,
    alpha: float = 0.1,
    gamma: float = 0.99,
    initial_epsilon: float = 1.0,
    min_epsilon: float = 0.01,
    epsilon_decay: float = 0.9999,
    episodes: int = 10000,
    eval_every: int = 1000,
    eval_episodes: int = 100,
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

    print("Q-learning Training completed.")
    return q_table, rewards, lengths, epsilons, time_steps


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
    return q_table_a + q_table_b, rewards, lengths, epsilons, time_steps


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
    print("Speedy Q-learning Training Completed.")
    return Q_k, rewards, lengths, epsilons, time_steps


def ultimate_sdql_training(env, epsilon=1, min_epsilon=0.1, epsilon_decay=0.999999, T=100000, gamma=0.99, eval_interval=10000, eval_episodes=1000):
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
    print("Speedy Double Q-learning Training Done!")
    return Q_A + Q_B, rewards, lengths, epsilons, time_steps

env = gym.make('FrozenLake-v1', is_slippery=False)

# Define the algorithms and their corresponding functions
algorithms = [
    (train_q_learning, {"episodes": 10000}),
    (train_double_q_learning, {"episodes": 10000}),
    (asynchronous_speedy_q_learning, {"total_time_steps": 10000}),
    (ultimate_sdql_training, {"T": 100000})
]

# Titles for the plots
titles = ["Q-Learning", "Double Q-Learning", "SQL", "Speedy Double Q-Learning"]

# Run the algorithms and collect results
results = []
for algorithm, params in algorithms:
    rewards, lengths, epsilons, time_steps = run_algorithm(env, algorithm, **params)
    results.append((rewards, lengths, epsilons, time_steps))

# Compare the algorithms
compare_algorithms(results, titles, window_size=100)
