import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from typing import List, Tuple
import gym_simplegrid

# Define a function to run a given algorithm
def run_algorithm(env, algorithm_func, **kwargs):
    results = algorithm_func(env, **kwargs)
    if isinstance(results, tuple):
        return results[-4:]  # Return rewards, lengths, epsilons, and time_steps
    else:
        raise ValueError("Algorithm function did not return expected output")

def moving_average(data: List[float], window_size: int) -> List[float]:
    if len(data) < window_size:
        return np.array(data)  # Return the data as is if it's shorter than the window size
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def initialize_random_q_table(env, goal_state=15):
    q_table = np.random.uniform(low=0, high=0.1, size=(env.observation_space.n, env.action_space.n))
    q_table[goal_state, :] = 0  # Ensure the goal state has Q-values of 0
    return q_table

def epsilon_greedy_policy(Q, state, epsilon, env):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state, :])


def evaluate_policy(env, q_table, episodes=10) -> Tuple[float, float]:
    """Evaluate the Q-learning agent for a certain number of episodes and return average reward and steps."""
    total_reward, total_length = 0, 0
    options = {
        'start_loc': 0,
        'goal_loc': 15
    }

    for _ in range(episodes):
        state = env.reset(seed=1234, options=options)[0]
        done = truncated = False
        episode_reward, steps = 0, 0

        while not (done):
            action = np.argmax(q_table[state])
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            if steps > 500:
                print("eval broke")
                break
            

        total_reward += episode_reward
        total_length += steps

    avg_reward = total_reward / episodes
    avg_length = total_length / episodes
    return avg_reward, avg_length

def sdql(env, epsilon=1, min_epsilon=0.1, epsilon_decay=0.999999, T=100000, gamma=0.99, eval_interval=50, eval_episodes=1):
    state_space = env.observation_space.n
    action_space = env.action_space.n

    Q_A = initialize_random_q_table(env)
    Q_B = initialize_random_q_table(env)
    Q_A_minus_1, Q_B_minus_1 = np.copy(Q_A), np.copy(Q_B)

    N_A = np.zeros((state_space, action_space))
    N_B = np.zeros((state_space, action_space))

    rewards, lengths, epsilons, time_steps= [], [], [], []

    indices = [5, 7, 11, 12]
    mask_hole_indeces = [True if i not in indices else False for i in range(16)]
    
    k_A = k_B = t = 0
    alpha_A = alpha_B = 1

    options={'start_loc':0, 'goal_loc':15}

    state = env.reset(seed=1234, options=options)[0]

    first = True
    while t <= T:
        if np.random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            combined_Q = (Q_A[state, :] + Q_B[state, :]) / 2
            action = np.argmax(combined_Q)

        next_state, reward, done, truncated, info = env.step(action)

        if np.random.uniform(0,1) < 0.5:  # Update Q_A
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
            T_kQ_B_minus_1 = (1 - eta) * Q_B_minus_1[state, action] + eta * (
                        reward + gamma * Q_A_minus_1[next_state, a_star])
            T_kQ_B = (1 - eta) * Q_B[state, action] + eta * (reward + gamma * Q_A[next_state, a_plus])

            Q_B_minus_1[state, action] = Q_B[state, action]
            Q_B[state, action] = (1 - alpha_B) * Q_B[state, action] + alpha_B * (
                        k_B * T_kQ_B - (k_B - 1) * T_kQ_B_minus_1)
            N_B[state, action] += 1

        state = next_state

        # Update counters and learning rates if needed
        if np.min(N_A[mask_hole_indeces]) > 0:
            print("N_A got reset")
            k_A += 1
            alpha_A = 1 / (k_A + 1)
            N_A.fill(0)  # Reset visit counts for A
        

        if np.min(N_B[mask_hole_indeces]) > 0:
            print("N_B got reset")
            k_B += 1
            alpha_B = 1 / (k_B + 1)
            N_B.fill(0)  # Reset visit counts for B

        epsilon = max(min_epsilon, epsilon * epsilon_decay)  # Reduce epsilon
        t += 1
        if done:
            state = env.reset(seed=1234, options=options)[0]

        if (t + 1) % eval_interval == 0:
            avg_reward, avg_length = evaluate_policy(env, Q_A+Q_B, eval_episodes)
            if first and avg_reward == 1:
                first = False
                print(f"Time Step = {t + 1}: Avg Reward = {avg_reward}, Avg Length = {avg_length}, epsilon = {epsilon}")
            time_steps.append(t+1)
            rewards.append(avg_reward)
            lengths.append(avg_length)
            epsilons.append(epsilon)

    print("SDQL training completed.")

    return Q_A, rewards, lengths, epsilons, time_steps


def speedy_q_learning(
        env, alpha=1, epsilon=1.0, gamma=0.99, total_time_steps=10000,
        eval_every: int = 100,
        eval_episodes: int = 10,
        epsilon_decay=0.99999,
        min_epsilon = 0.1,
        goal_state=15
):
    Q_k = initialize_random_q_table(env, goal_state)
    Q_k_minus_1 = np.copy(Q_k)

    state_space = env.observation_space.n
    action_space = env.action_space.n
    t = k = 0  # Iteration counter
    N = np.zeros((state_space, action_space), dtype=int)

    state = env.reset(seed=1234, options={'start_loc':0, 'goal_loc':15})[0]
    rewards, lengths, epsilons, total_steps = [], [], [], []
    mask_hole_indeces = [True if i not in [5, 7, 11, 12] else False for i in range(16)]

    first = True
    # hole_indeces = [19,29,35,41,42,46,49,52,54,59]

    while t <= total_time_steps:
        action = epsilon_greedy_policy(Q_k, state, epsilon, env)
        next_state, reward, done, _, _ = env.step(action)
        
        eta = 1 / (N[state][action] + 1)

        # Compute temporal differences
        best_next_action_k_minus_1 = np.argmax(Q_k_minus_1[next_state, :])
        best_next_action_k = np.argmax(Q_k[next_state, :])
        T_kQ_k_minus_1 = (1 - eta) * Q_k_minus_1[state, action] + eta * (reward + gamma * Q_k_minus_1[next_state, best_next_action_k_minus_1])
        T_kQ_k = (1 - eta) * Q_k[state, action] + eta * (reward + gamma * Q_k[next_state, best_next_action_k])

        # Update Q_k+1
        Q_k_plus_1 = (1 - alpha) * Q_k[state, action] + alpha * (k * T_kQ_k - (k-1) * T_kQ_k_minus_1) # Update rule

        # Update Q-table references
        Q_k_minus_1 = np.copy(Q_k)
        Q_k[state, action] = np.copy(Q_k_plus_1)

        # Move to next state
        N[state][action] += 1
        # print( N[state][action])
        state = next_state

        # Check if all state-action pairs have been visited
        
        if np.min(N[mask_hole_indeces]) > 0:
            print("N got reset")
            k += 1
            alpha = 1 / (k + 1)
            N = np.zeros_like(N)  # Reset visit counts

            # check kon ke satisfy shode ya na
            # another function to find a subset to be greater than the number of possible accessible state

        t += 1

        epsilon = max(min_epsilon, epsilon_decay * epsilon)

        if done:
            state = env.reset(options={'start_loc':0, 'goal_loc':15})[0]

        if (t + 1) % eval_every == 0:
            avg_reward, avg_length = evaluate_policy(env, Q_k, eval_episodes)
            if first and avg_reward == 1:
                first = False
                print(f"Total time step: {t + 1}, Avg. Reward: {avg_reward}, Avg. Length: {avg_length}, Epsilon: {epsilon}")
            total_steps.append(t)
            rewards.append(avg_reward)
            lengths.append(avg_length)
            epsilons.append(epsilon)
    print("SQL training completed.")
    return Q_k, rewards, lengths, epsilons, total_steps


def double_q_learning(
    env: gym.Env,
    alpha: float = 0.1,
    gamma: float = 0.99,
    initial_epsilon: float = 1.0,
    min_epsilon: float = 0.01,
    epsilon_decay: float = 0.9999,
    episodes: int = 20000,
    eval_every: int = 100,
    eval_episodes: int = 20,
    goal_state=15
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
    # n_states = env.observation_space.n
    # n_actions = env.action_space.n
    # q_table_a = np.random.uniform(low=0, high=0.1, size=(n_states, n_actions))
    # q_table_b = np.random.uniform(low=0, high=0.1, size=(n_states, n_actions))
    # q_table_a[
    #     (env.desc == b"G").flatten()
    # ] = 0  # Assuming 'G' is the goal/terminal state
    # q_table_b[
    #     (env.desc == b"G").flatten()
    # ] = 0  # Assuming 'G' is the goal/terminal state

    q_table_a = initialize_random_q_table(env, goal_state)
    q_table_b = initialize_random_q_table(env, goal_state)

    epsilon = initial_epsilon
    rewards, lengths, epsilons, total_steps = [], [], [],[]
    env.reset(seed=1234, options={'start_loc':0, 'goal_loc':15})
    max_steps = 1000
    first = True
    steps = 0
    
    for episode in range(1, episodes+1):
        state = env.reset(options={'start_loc':0, 'goal_loc':15})[0]
        done = False
        
        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table_a[state, :] + q_table_b[state, :])

            next_state, reward, done, truncated, info = env.step(action)
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
            steps += 1
            state = next_state
            if steps % eval_every == 0:
                avg_reward, avg_length = evaluate_policy(
                    env, q_table_a, eval_episodes
                )
                if first and avg_reward == 1:
                    first = False
                    print(f"TotalTime steps: {steps}, Episode: {episode + 1}, Avg. Reward: {avg_reward}, Avg. Length: {avg_length}, Epsilon: {epsilon}")
                total_steps.append(steps)
                rewards.append(avg_reward)
                lengths.append(avg_length)
                epsilons.append(epsilon)
            # # If done (if we're dead) : finish episode
            # if done:
            #     break
            # if reward == 1:
            #     print("HOORAY WE GOT THERE!!!")
            
        epsilon = max(min_epsilon, epsilon_decay * epsilon)
        # print(len(seq))


    print("Double Q-learning training completed.")
    return q_table_a, q_table_b, rewards, lengths, epsilons, total_steps



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
        'goal_loc': 15
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

    print("QL Training completed.")
    return q_table, rewards, lengths, epsilons, total_steps

def compare_algorithms(results, titles, save_path=None):
    plt.figure(figsize=(18, 6))

    colors = ['b', 'g', 'r', 'purple']

    # Plot Average Cumulative Rewards
    plt.subplot(1, 3, 1)
    for i, (rewards, _, _, time_steps) in enumerate(results):
        plt.plot(time_steps, rewards, color=colors[i % len(colors)], label=f'{titles[i]} Reward')
    plt.title("Average Cumulative Reward")
    plt.xlabel("Time Steps")
    plt.ylabel("Average Cumulative Reward")
    plt.legend()

    # Plot Average Steps
    plt.subplot(1, 3, 2)
    for i, (_, lengths, _, time_steps) in enumerate(results):
        plt.plot(time_steps, lengths, color=colors[i % len(colors)], label=f'{titles[i]} Steps')
    plt.title("Average Steps")
    plt.xlabel("Time Steps")
    plt.ylabel("Average Steps")
    plt.legend()

    # Plot Epsilon Decay
    plt.subplot(1, 3, 3)
    for i, (_, _, epsilons, time_steps) in enumerate(results):
        plt.plot(time_steps, epsilons, color=colors[i % len(colors)], label=f'{titles[i]} Epsilon')
    plt.title("Epsilon Decay")
    plt.xlabel("Time Steps")
    plt.ylabel("Epsilon Value")
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plots saved to {save_path}")
    else:
        plt.show()


def moving_average(data: List[float], window_size: int) -> List[float]:
    if len(data) < window_size:
        return np.array(data)  # Return the data as is if it's shorter than the window size
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


# def pad_sequences(sequences, pad_value=0.0):
#     max_len = max(len(seq) for seq in sequences)
#     padded_sequences = np.array([np.pad(seq, (0, max_len - len(seq)), 'constant', constant_values=pad_value) for seq in sequences])
#     return padded_sequences

# def pad_and_extend_sequences(sequences, max_length=10000):
#     max_len = max(max(len(seq), max_length) for seq in sequences)
#     padded_sequences = np.array([np.pad(seq, (0, max_len - len(seq)), 'edge') for seq in sequences])
#     return padded_sequences

# def extend_time_steps(time_steps, target_length=10000):
#     last_value = time_steps[-1]
#     extended_steps = np.arange(last_value + 1, last_value + (target_length - len(time_steps)) + 1)
#     extended_time_steps = np.concatenate([time_steps, extended_steps])
#     return extended_time_steps
    


def run_multiple_times(env, algorithm_func, params, n_runs=3):
    all_rewards = []
    all_lengths = []
    all_epsilons = []
    all_time_steps = []

    max_length = 10000  # The target length for padding

    for _ in range(n_runs):
        rewards, lengths, epsilons, time_steps = run_algorithm(env, algorithm_func, **params)
        
        # Pad each array to the target length
        rewards = pad_to_length(rewards, max_length)
        lengths = pad_to_length(lengths, max_length)
        epsilons = pad_to_length(epsilons, max_length)
        time_steps = pad_to_length(time_steps, max_length)

        all_rewards.append(rewards)
        all_lengths.append(lengths)
        all_epsilons.append(epsilons)
        all_time_steps.append(time_steps)

    # Compute the average across runs
    avg_rewards = np.mean(all_rewards, axis=0)
    avg_lengths = np.mean(all_lengths, axis=0)
    avg_epsilons = np.mean(all_epsilons, axis=0)
    avg_time_steps = np.mean(all_time_steps, axis=0)

    return avg_rewards, avg_lengths, avg_epsilons, avg_time_steps


# Define the SimpleGrid environment
obstacle_map = [
    "0000",
    "0101",
    "0001",
    "1000",
]
env = gym.make('SimpleGrid-4x4-v0', render_mode='rgb_array', obstacle_map=obstacle_map)

alpha = 0.1
gamma = 0.99
epsilon = 1.0
initial_epsolon=1.0
epsilon_decay = 0.9999
min_epsilon = 0.01
episodes = 100
total_time_steps = 10000
eval_episodes=1
eval_every=500

# q_learning(env, alpha=alpha, gamma=gamma, initial_epsilon=initial_epsolon, min_epsilon=min_epsilon, epsilon_decay=epsilon_decay, episodes=episodes, eval_every=eval_every, eval_episodes=eval_episodes)
# Define the algorithms and their corresponding functions
algorithms = [
    (q_learning, {"episodes": episodes, "alpha": alpha, "gamma": gamma, "initial_epsilon": initial_epsolon, "min_epsilon": min_epsilon, "epsilon_decay": epsilon_decay, "eval_every": eval_every, "eval_episodes": eval_episodes}),
    (double_q_learning, {"episodes": episodes, "alpha": alpha, "gamma": gamma, "initial_epsilon": initial_epsolon, "min_epsilon": min_epsilon, "epsilon_decay": epsilon_decay, "eval_every": eval_every, "eval_episodes": eval_episodes}),
    (speedy_q_learning, {"total_time_steps": total_time_steps, "alpha": alpha, "gamma": gamma, "epsilon": epsilon, "min_epsilon": min_epsilon, "epsilon_decay": 0.999999, "eval_every": eval_every, "eval_episodes": eval_episodes}),
    (sdql, {"T": total_time_steps, "epsilon": epsilon, "gamma": gamma, "min_epsilon": min_epsilon, "epsilon_decay": 0.9999995, "eval_interval": eval_every, "eval_episodes": eval_episodes})
]

# Titles for the plots
titles = ["Q-Learning", "Double Q-Learning", "Speedy Q-Learning", "SDQL"]

# Run the algorithms and collect results
results = []
for algorithm, params in algorithms:
    avg_rewards, avg_lengths, avg_epsilons, avg_time_steps = run_multiple_times(env, algorithm, params, n_runs=3)
    results.append((avg_rewards, avg_lengths, avg_epsilons, avg_time_steps))

compare_algorithms(results, titles)
