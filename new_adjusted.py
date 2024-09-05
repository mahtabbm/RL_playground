import numpy as np
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from typing import List, Tuple
import matplotlib.pyplot as plt


def initialize_random_q_table(env):
    q_table = np.random.uniform(low=0, high=0.1, size=(env.observation_space.n, env.action_space.n))
    q_table[(env.desc == b"G").flatten()] = 0  # Ensure the goal state has Q-values of 0
    return q_table

def epsilon_greedy_policy(Q, state, epsilon, env):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state, :])

def train_q_learning_multiple_runs(
    env: gym.Env,
    alpha: float = 0.1,
    gamma: float = 0.99,
    initial_epsilon: float = 1.0,
    min_epsilon: float = 0.01,
    epsilon_decay: float = 0.9999,
    episodes: int = 10000,
    num_runs: int = 10,
    seeds: List[int] = None
) -> Tuple:
    """Trains an agent using the Q-learning algorithm on a specified environment and performs multiple runs for statistical analysis."""

    if seeds is None:
        seeds = [np.random.randint(0, 10000) for _ in range(num_runs)]

    run_rewards = []  # To store cumulative rewards for each run
    run_lengths = []  # To store cumulative episode lengths for each run
    run_epsilons = []  # To store epsilon values for each run
    run_time_steps = []  # To store time steps for each run
    
    for run_idx in range(num_runs):
        print(f"Run {run_idx + 1} / {num_runs} with seed {seeds[run_idx]}")
        
        # Reset environment and Q-table for each run
        q_table = initialize_random_q_table(env)
        env.reset(seed=seeds[run_idx])
        epsilon = initial_epsilon
        total_steps = 0
        training_rewards = []
        episode_lengths = []  # To track episode lengths (number of steps per episode)
        epsilons = []  # To track epsilon values over time

        for episode in range(episodes):
            state = env.reset(seed=seeds[run_idx])[0]
            done = False
            episode_reward = 0
            episode_length = 0  # Track the length of the current episode

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
                episode_reward += reward
                episode_length += 1  # Increment the length of the current episode

                # Track epsilon at this time step
                epsilons.append(epsilon)

            # Track rewards and episode lengths for this run
            training_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Epsilon decay
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # After all episodes, append cumulative rewards, episode lengths, epsilons, and time steps
        cumulative_rewards = np.cumsum(training_rewards)
        cumulative_lengths = np.cumsum(episode_lengths)
        
        run_rewards.append(cumulative_rewards)
        run_lengths.append(cumulative_lengths)
        run_epsilons.append(epsilons[:len(cumulative_rewards)])  # Match the length of rewards and epsilons
        run_time_steps.append(list(range(1, len(cumulative_rewards) + 1)))

    # Compute average and standard deviation of rewards, lengths, and epsilons across runs at each time step
    max_time_steps = min(len(r) for r in run_rewards)  # Ensure we use the shortest run length for analysis

    # Convert the rewards, lengths, and epsilons to matrices
    rewards_matrix = np.array([run[:max_time_steps] for run in run_rewards])
    lengths_matrix = np.array([run[:max_time_steps] for run in run_lengths])
    epsilons_matrix = np.array([run[:max_time_steps] for run in run_epsilons])

    # Calculate average and standard deviation for rewards, episode lengths, and epsilons
    avg_rewards = np.mean(rewards_matrix, axis=0)
    std_rewards = np.std(rewards_matrix, axis=0)
    avg_lengths = np.mean(lengths_matrix, axis=0)
    std_lengths = np.std(lengths_matrix, axis=0)
    avg_epsilons = np.mean(epsilons_matrix, axis=0)
    std_epsilons = np.std(epsilons_matrix, axis=0)

    time_steps = run_time_steps[0][:max_time_steps]  # Take time steps from any run (since they're the same length)

    print("Q-learning multiple runs completed.")
    return avg_rewards, std_rewards, avg_lengths, std_lengths, avg_epsilons, std_epsilons, time_steps


def train_double_q_learning_multiple_runs(
    env: gym.Env,
    alpha: float = 0.1,
    gamma: float = 0.99,
    initial_epsilon: float = 1.0,
    min_epsilon: float = 0.01,
    epsilon_decay: float = 0.9999,
    episodes: int = 10000,
    eval_every: int = 1000,
    eval_episodes: int = 100,
    num_runs: int = 10,
    seeds: List[int] = None
) -> Tuple:
    """
    Trains an agent using the Double Q-learning algorithm on a specified environment and performs multiple runs.

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
        num_runs (int): Number of runs for statistical analysis.
        seeds (List[int]): List of seeds for reproducibility. Each run uses a different seed.

    Returns:
        A tuple containing:
        - Average cumulative rewards across all runs.
        - Standard deviation of rewards.
        - Average episode lengths.
        - Standard deviation of episode lengths.
        - Average epsilon values.
        - Standard deviation of epsilon values.
        - Time steps.
    """

    if seeds is None:
        seeds = [np.random.randint(0, 10000) for _ in range(num_runs)]

    run_rewards = []  # To store cumulative rewards for each run
    run_lengths = []  # To store cumulative episode lengths for each run
    run_epsilons = []  # To store epsilon values for each run
    run_time_steps = []  # To store time steps for each run

    for run_idx in range(num_runs):
        print(f"Run {run_idx + 1} / {num_runs} with seed {seeds[run_idx]}")
        
        # Reset environment and Q-tables for each run
        q_table_a = initialize_random_q_table(env)
        q_table_b = initialize_random_q_table(env)
        env.reset(seed=seeds[run_idx])
        epsilon = initial_epsilon
        total_steps = 0
        training_rewards = []
        episode_lengths = []  # To track episode lengths (number of steps per episode)
        epsilons = []  # To track epsilon values over time

        for episode in range(episodes):
            state = env.reset(seed=seeds[run_idx])[0]
            done = False
            episode_reward = 0
            episode_length = 0  # Track the length of the current episode

            while not done:
                # Epsilon-greedy action selection using the average of Q-tables A and B
                action = epsilon_greedy_policy(Q=q_table_a + q_table_b / 2, state=state, epsilon=epsilon, env=env)

                next_state, reward, done, _, _ = env.step(action)

                if np.random.uniform(0, 1) < 0.5:
                    best_next_action = np.argmax(q_table_a[next_state, :])
                    td_target = reward + gamma * q_table_b[next_state, best_next_action]
                    q_table_a[state, action] += alpha * (td_target - q_table_a[state, action])
                else:
                    best_next_action = np.argmax(q_table_b[next_state, :])
                    td_target = reward + gamma * q_table_a[next_state, best_next_action]
                    q_table_b[state, action] += alpha * (td_target - q_table_b[state, action])

                total_steps += 1
                episode_reward += reward
                episode_length += 1  # Increment the length of the current episode

                # Track epsilon at this time step
                epsilons.append(epsilon)

            # Track rewards and episode lengths for this run
            training_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Epsilon decay
            epsilon = max(min_epsilon, epsilon_decay * epsilon)

        # After all episodes, append cumulative rewards, episode lengths, epsilons, and time steps
        cumulative_rewards = np.cumsum(training_rewards)
        cumulative_lengths = np.cumsum(episode_lengths)

        run_rewards.append(cumulative_rewards)
        run_lengths.append(cumulative_lengths)
        run_epsilons.append(epsilons[:len(cumulative_rewards)])  # Match the length of rewards and epsilons
        run_time_steps.append(list(range(1, len(cumulative_rewards) + 1)))

    # Compute average and standard deviation of rewards, lengths, and epsilons across runs at each time step
    max_time_steps = min(len(r) for r in run_rewards)  # Ensure we use the shortest run length for analysis

    # Convert the rewards, lengths, and epsilons to matrices
    rewards_matrix = np.array([run[:max_time_steps] for run in run_rewards])
    lengths_matrix = np.array([run[:max_time_steps] for run in run_lengths])
    epsilons_matrix = np.array([run[:max_time_steps] for run in run_epsilons])

    # Calculate average and standard deviation for rewards, episode lengths, and epsilons
    avg_rewards = np.mean(rewards_matrix, axis=0)
    std_rewards = np.std(rewards_matrix, axis=0)
    avg_lengths = np.mean(lengths_matrix, axis=0)
    std_lengths = np.std(lengths_matrix, axis=0)
    avg_epsilons = np.mean(epsilons_matrix, axis=0)
    std_epsilons = np.std(epsilons_matrix, axis=0)

    time_steps = run_time_steps[0][:max_time_steps]  # Take time steps from any run (since they're the same length)

    print("Double Q-learning multiple runs completed.")
    return avg_rewards, std_rewards, avg_lengths, std_lengths, avg_epsilons, std_epsilons, time_steps


def asynchronous_speedy_q_learning_multiple_runs(
    env, 
    epsilon=1.0, 
    gamma=0.99, 
    total_time_steps=50000,
    eval_every: int = 1000,
    eval_episodes: int = 100,         
    epsilon_decay=0.99999,
    min_epsilon=0.01,
    num_runs: int = 10,
    seeds: List[int] = None
) -> Tuple:
    """
    Trains an agent using the Asynchronous Speedy Q-learning algorithm over multiple runs for statistical analysis.

    Args:
        env: The environment to train the agent on.
        epsilon: Starting value for epsilon in the epsilon-greedy strategy.
        gamma: Discount factor for future rewards.
        total_time_steps: Total number of training time steps.
        eval_every: Frequency of evaluation phases during training.
        eval_episodes: Number of episodes to run during each evaluation phase.
        epsilon_decay: Rate at which epsilon decays after each step.
        min_epsilon: Minimum value that epsilon can decay to over time.
        num_runs: Number of runs for statistical analysis.
        seeds: List of seeds for reproducibility.

    Returns:
        A tuple containing:
        - Average cumulative rewards across all runs.
        - Standard deviation of rewards.
        - Average episode lengths.
        - Standard deviation of episode lengths.
        - Average epsilon values.
        - Standard deviation of epsilon values.
        - Time steps.
    """
    
    if seeds is None:
        seeds = [np.random.randint(0, 10000) for _ in range(num_runs)]

    run_rewards = []  # To store cumulative rewards for each run
    run_lengths = []  # To store cumulative episode lengths for each run
    run_epsilons = []  # To store epsilon values for each run
    run_time_steps = []  # To store time steps for each run

    for run_idx in range(num_runs):
        print(f"Run {run_idx + 1} / {num_runs} with seed {seeds[run_idx]}")
        
        state_space = env.observation_space.n
        action_space = env.action_space.n
        alpha = 1
        Q_k = initialize_random_q_table(env)
        Q_k_minus_1 = np.copy(Q_k)
        k = 0
        N = np.zeros((state_space, action_space), dtype=int)
        state = env.reset(seed=seeds[run_idx])[0]
        t = 0
        training_rewards = []
        episode_lengths = []
        epsilons = []

        lake_map = env.desc
        frozen_lake_binary = np.array((lake_map != b'H') & (lake_map != b'G')).astype(int).flatten()

        episode_reward = 0
        episode_length = 0

        while t <= total_time_steps:
            action = epsilon_greedy_policy(Q_k, state, epsilon, env)
            next_state, reward, done, _, _ = env.step(action)

            eta = 1 / (N[state][action] + 1)

            # Compute temporal differences
            best_next_action_k_minus_1 = np.argmax(Q_k_minus_1[next_state])
            best_next_action_k = np.argmax(Q_k[next_state])
            T_kQ_k_minus_1 = (1 - eta) * Q_k_minus_1[state, action] + eta * (
                reward + gamma * Q_k_minus_1[next_state, best_next_action_k_minus_1])
            T_kQ_k = (1 - eta) * Q_k[state, action] + eta * (
                reward + gamma * Q_k[next_state, best_next_action_k])

            # Update Q_k+1
            Q_k_plus_1 = (1 - alpha) * Q_k[state, action] + alpha * (k * T_kQ_k - (k - 1) * T_kQ_k_minus_1)
            N[state][action] += 1

            # Update Q-table references
            Q_k_minus_1 = np.copy(Q_k)
            Q_k[state, action] = np.copy(Q_k_plus_1)
            
            state = next_state

            episode_reward += reward
            episode_length += 1
            epsilons.append(epsilon)

            # Check if all state-action pairs have been visited
            if np.min(N[frozen_lake_binary == 1]) > 0:
                k += 1
                alpha = 1 / (k + 1)
                N.fill(0)  # Reset visit counts

            epsilon = max(min_epsilon, epsilon_decay * epsilon)
            t += 1

            if done:
                state = env.reset(seed=seeds[run_idx])[0]
                training_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_reward = 0
                episode_length = 0

            if (t + 1) % eval_every == 0 and episode_reward > 0:
                print(f"Time Step = {t + 1}: Avg Reward = {np.mean(training_rewards[-eval_every:])}, Avg Length = {np.mean(episode_lengths[-eval_every:])}, epsilon = {epsilon}")
        
        # After each run, append cumulative rewards, episode lengths, and epsilons
        cumulative_rewards = np.cumsum(training_rewards)
        cumulative_lengths = np.cumsum(episode_lengths)

        run_rewards.append(cumulative_rewards)
        run_lengths.append(cumulative_lengths)
        run_epsilons.append(epsilons[:len(cumulative_rewards)])  # Match the length of rewards and epsilons
        run_time_steps.append(list(range(1, len(cumulative_rewards) + 1)))

    # Compute average and standard deviation of rewards, lengths, and epsilons across runs at each time step
    max_time_steps = min(len(r) for r in run_rewards)

    rewards_matrix = np.array([run[:max_time_steps] for run in run_rewards])
    lengths_matrix = np.array([run[:max_time_steps] for run in run_lengths])
    epsilons_matrix = np.array([run[:max_time_steps] for run in run_epsilons])

    avg_rewards = np.mean(rewards_matrix, axis=0)
    std_rewards = np.std(rewards_matrix, axis=0)
    avg_lengths = np.mean(lengths_matrix, axis=0)
    std_lengths = np.std(lengths_matrix, axis=0)
    avg_epsilons = np.mean(epsilons_matrix, axis=0)
    std_epsilons = np.std(epsilons_matrix, axis=0)

    time_steps = run_time_steps[0][:max_time_steps]

    print("Asynchronous Speedy Q-learning multiple runs completed.")
    return avg_rewards, std_rewards, avg_lengths, std_lengths, avg_epsilons, std_epsilons, time_steps

def ultimate_sdql_training_multiple_runs(
    env,
    epsilon=1,
    min_epsilon=0.1,
    epsilon_decay=0.999999,
    T=50000,
    gamma=0.99,
    eval_interval=50,
    eval_episodes=1,
    num_runs: int = 10,
    seeds: List[int] = None
) -> Tuple:
    """
    Trains an agent using the Ultimate Speedy Double Q-learning (SDQL) algorithm over multiple runs for statistical analysis.

    Args:
        env: The environment to train the agent on.
        epsilon: Starting value for epsilon in the epsilon-greedy strategy.
        min_epsilon: Minimum value that epsilon can decay to.
        epsilon_decay: Rate at which epsilon decays after each step.
        T: Total number of time steps.
        gamma: Discount factor for future rewards.
        eval_interval: Frequency of evaluation during training.
        eval_episodes: Number of episodes to run during evaluation.
        num_runs: Number of runs for statistical analysis.
        seeds: List of seeds for reproducibility.

    Returns:
        A tuple containing:
        - Average cumulative rewards across all runs.
        - Standard deviation of rewards.
        - Average episode lengths.
        - Standard deviation of episode lengths.
        - Average epsilon values.
        - Standard deviation of epsilon values.
        - Time steps.
    """

    if seeds is None:
        seeds = [np.random.randint(0, 10000) for _ in range(num_runs)]

    run_rewards = []  # To store cumulative rewards for each run
    run_lengths = []  # To store cumulative episode lengths for each run
    run_epsilons = []  # To store epsilon values for each run
    run_time_steps = []  # To store time steps for each run

    for run_idx in range(num_runs):
        print(f"Run {run_idx + 1} / {num_runs} with seed {seeds[run_idx]}")

        state_space = env.observation_space.n
        action_space = env.action_space.n

        Q_A = initialize_random_q_table(env)
        Q_B = initialize_random_q_table(env)
        Q_A_minus_1, Q_B_minus_1 = np.copy(Q_A), np.copy(Q_B)

        N_A = np.zeros((state_space, action_space))
        N_B = np.zeros((state_space, action_space))

        k_A = k_B = t = 0
        alpha_A = alpha_B = 1

        state = env.reset(seed=seeds[run_idx])[0]

        lake_map = env.desc
        frozen_lake_binary = np.array((lake_map != b'H') & (lake_map != b'G')).astype(int).flatten()

        episode_reward = 0
        episode_length = 0
        epsilons = []

        training_rewards = []
        episode_lengths = []

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

                T_kQ_A_minus_1 = (1 - eta) * Q_A_minus_1[state, action] + eta * (
                    reward + gamma * Q_B_minus_1[next_state, a_star])
                T_kQ_A = (1 - eta) * Q_A[state, action] + eta * (
                    reward + gamma * Q_B[next_state, a_plus])

                Q_A_minus_1[state, action] = Q_A[state, action]
                Q_A[state, action] = (1 - alpha_A) * Q_A[state, action] + alpha_A * (
                    k_A * T_kQ_A - (k_A - 1) * T_kQ_A_minus_1)
                N_A[state, action] += 1
            else:  # Update Q_B
                eta = 1 / (N_B[state, action] + 1)
                a_star = np.argmax(Q_B_minus_1[next_state, :])
                a_plus = np.argmax(Q_B[next_state, :])
                T_kQ_B_minus_1 = (1 - eta) * Q_B_minus_1[state, action] + eta * (
                    reward + gamma * Q_A_minus_1[next_state, a_star])
                T_kQ_B = (1 - eta) * Q_B[state, action] + eta * (
                    reward + gamma * Q_A[next_state, a_plus])

                Q_B_minus_1[state, action] = Q_B[state, action]
                Q_B[state, action] = (1 - alpha_B) * Q_B[state, action] + alpha_B * (
                    k_B * T_kQ_B - (k_B - 1) * T_kQ_B_minus_1)
                N_B[state, action] += 1

            state = next_state

            episode_reward += reward
            episode_length += 1
            epsilons.append(epsilon)

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
                state = env.reset(seed=seeds[run_idx])[0]
                training_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_reward = 0
                episode_length = 0

            if (t + 1) % eval_interval == 0 and episode_reward > 0:
                print(f"Time Step = {t + 1}: Avg Reward = {np.mean(training_rewards[-eval_interval:])}, "
                      f"Avg Length = {np.mean(episode_lengths[-eval_interval:])}, epsilon = {epsilon}")

        # After each run, append cumulative rewards, episode lengths, and epsilons
        cumulative_rewards = np.cumsum(training_rewards)
        cumulative_lengths = np.cumsum(episode_lengths)

        run_rewards.append(cumulative_rewards)
        run_lengths.append(cumulative_lengths)
        run_epsilons.append(epsilons[:len(cumulative_rewards)])  # Match the length of rewards and epsilons
        run_time_steps.append(list(range(1, len(cumulative_rewards) + 1)))

    # Compute average and standard deviation of rewards, lengths, and epsilons across runs at each time step
    max_time_steps = min(len(r) for r in run_rewards)

    rewards_matrix = np.array([run[:max_time_steps] for run in run_rewards])
    lengths_matrix = np.array([run[:max_time_steps] for run in run_lengths])
    epsilons_matrix = np.array([run[:max_time_steps] for run in run_epsilons])

    avg_rewards = np.mean(rewards_matrix, axis=0)
    std_rewards = np.std(rewards_matrix, axis=0)
    avg_lengths = np.mean(lengths_matrix, axis=0)
    std_lengths = np.std(lengths_matrix, axis=0)
    avg_epsilons = np.mean(epsilons_matrix, axis=0)
    std_epsilons = np.std(epsilons_matrix, axis=0)

    time_steps = run_time_steps[0][:max_time_steps]

    print("Ultimate SDQL multiple runs completed.")
    return avg_rewards, std_rewards, avg_lengths, std_lengths, avg_epsilons, std_epsilons, time_steps

def plot_comparison(
    time_steps,
    avg_rewards, std_rewards,
    avg_lengths, std_lengths,
    avg_epsilons, std_epsilons,
    algorithm_name="Algorithm"
):
    """Plots average cumulative rewards, episode lengths, and epsilon values vs. time steps."""
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot 1: Average (and STD) cumulative reward vs. time steps
    axes[0].plot(time_steps, avg_rewards, label=f"{algorithm_name} - Avg. Reward")
    axes[0].fill_between(time_steps, avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.3)
    axes[0].set_title(f"{algorithm_name}: Cumulative Reward vs Time Steps")
    axes[0].set_xlabel("Time Steps (Samples)")
    axes[0].set_ylabel("Cumulative Reward")
    axes[0].legend()
    
    # Plot 2: Average (and STD) episode length vs. time steps
    axes[1].plot(time_steps, avg_lengths, label=f"{algorithm_name} - Avg. Episode Length", color='orange')
    axes[1].fill_between(time_steps, avg_lengths - std_lengths, avg_lengths + std_lengths, alpha=0.3, color='orange')
    axes[1].set_title(f"{algorithm_name}: Episode Length vs Time Steps")
    axes[1].set_xlabel("Time Steps (Samples)")
    axes[1].set_ylabel("Episode Length")
    axes[1].legend()

    # Plot 3: Epsilon vs. time steps
    axes[2].plot(time_steps, avg_epsilons, label=f"{algorithm_name} - Epsilon", color='green')
    axes[2].fill_between(time_steps, avg_epsilons - std_epsilons, avg_epsilons + std_epsilons, alpha=0.3, color='green')
    axes[2].set_title(f"{algorithm_name}: Epsilon vs Time Steps")
    axes[2].set_xlabel("Time Steps (Samples)")
    axes[2].set_ylabel("Epsilon")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


env = gym.make('FrozenLake-v1', is_slippery=False)

q_avg_rewards, q_std_rewards, q_avg_lengths, q_std_lengths, q_avg_epsilons, q_std_epsilons, q_time_steps = train_q_learning_multiple_runs(env, num_runs=3)
dql_avg_rewards, dql_std_rewards, dql_avg_lengths, dql_std_lengths, dql_avg_epsilons, dql_std_epsilons, dql_time_steps = train_double_q_learning_multiple_runs(env, num_runs=3)
sql_avg_rewards, sql_std_rewards, sql_avg_lengths, sql_std_lengths, sql_avg_epsilons, sql_std_epsilons, sql_time_steps = asynchronous_speedy_q_learning_multiple_runs(env, num_runs=3)
sdql_avg_rewards, sdql_std_rewards, sdql_avg_lengths, sdql_std_lengths, sdql_avg_epsilons, sdql_std_epsilons, sdql_time_steps = ultimate_sdql_training_multiple_runs(env, num_runs=3)


def plot_comparison_multiple_algorithms(
    algorithms_data: List[Tuple[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    metric: str = "rewards",
    ylabel: str = "Value",
    title: str = "Comparison of Algorithms",
    xlabel: str = "Time Steps (Samples)"
):
    """Compare multiple algorithms on a specific metric (rewards, lengths, or epsilons) in a single plot."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for alg_name, data in algorithms_data:
        time_steps, avg_values, std_values = data
        
        ax.plot(time_steps, avg_values, label=f"{alg_name} - Avg. {metric.capitalize()}")
        ax.fill_between(time_steps, avg_values - std_values, avg_values + std_values, alpha=0.2)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.show()

algorithms_data_rewards = [
    ("Q-learning", (q_time_steps, q_avg_rewards, q_std_rewards)),
    ("DQL", (dql_time_steps, dql_avg_rewards, dql_std_rewards)),
    ("SQL", (sql_time_steps, sql_avg_rewards, sql_std_rewards)),
    ("SDQL", (sdql_time_steps, sdql_avg_rewards, sdql_std_rewards)),
]

algorithms_data_lengths = [
    ("Q-learning", (q_time_steps, q_avg_lengths, q_std_lengths)),
    ("DQL", (dql_time_steps, dql_avg_lengths, dql_std_lengths)),
    ("SQL", (sql_time_steps, sql_avg_lengths, sql_std_lengths)),
    ("SDQL", (sdql_time_steps, sdql_avg_lengths, sdql_std_lengths)),
]

algorithms_data_epsilons = [
    ("Q-learning", (q_time_steps, q_avg_epsilons, q_std_epsilons)),
    ("DQL", (dql_time_steps, dql_avg_epsilons, dql_std_epsilons)),
    ("SQL", (sql_time_steps, sql_avg_epsilons, sql_std_epsilons)),
    ("SDQL", (sdql_time_steps, sdql_avg_epsilons, sdql_std_epsilons)),
]


# Plot for rewards
plot_comparison_multiple_algorithms(
    algorithms_data_rewards, 
    metric="rewards", 
    ylabel="Cumulative Reward",
    title="Cumulative Reward Across Algorithms"
)

# Plot for episode lengths
plot_comparison_multiple_algorithms(
    algorithms_data_lengths, 
    metric="lengths", 
    ylabel="Episode Length",
    title="Episode Length Across Algorithms"
)

# Plot for epsilons
plot_comparison_multiple_algorithms(
    algorithms_data_epsilons, 
    metric="epsilons", 
    ylabel="Epsilon",
    title="Epsilon Across Algorithms"
)