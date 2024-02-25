import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import List, Tuple


def train_q_learing(
    env: gym.Env,
    alpha: float = 0.1,
    gamma: float = 0.99,
    initial_epsilon: float = 1.0,
    min_epsilon: float = 0.01,
    epsilon_decay: float = 0.995,
    episodes: int = 20000,
    eval_every: int = 100,
    eval_episodes: int = 10,
) -> Tuple:
    """Trains an agent using the Q-learning algorithm on a specified environment.

    This function initializes a Q-table with random values and iteratively updates it based on the agent's experiences in the environment. The exploration rate (epsilon) decreases over time, allowing the agent to transition from exploring the environment to exploiting the learned Q-values. The function periodically evaluates the agent's performance using the current Q-table and returns the training history.

    Args:
        env (gym.Env): The environment to train the agent on. Must be compatible with the OpenAI Gym interface.
        alpha (float): The learning rate, determining how much of the new Q-value estimate to use. Defaults to 0.1.
        gamma (float): The discount factor, used to balance immediate and future rewards. Defaults to 0.99.
        initial_epsilon (float): The initial exploration rate, determining how often the agent explores random actions. Defaults to 1.0.
        min_epsilon (float): The minimum exploration rate after decay. Defaults to 0.01.
        epsilon_decay (float): The factor used for exponential decay of epsilon. Defaults to 0.995.
        episodes (int): The total number of episodes to train the agent for. Defaults to 10000.
        eval_every (int): The frequency (in episodes) at which to evaluate the agent's performance. Defaults to 100.
        eval_episodes (int): The number of episodes to use for each evaluation. Defaults to 10.

    Returns:
        tuple: A tuple containing three elements:
            - np.ndarray: The final Q-table learned by the agent.
            - list: A history of average rewards obtained by the agent during evaluation periods.
            - list: A history of average step lengths taken by the agent during evaluation periods.

    """
    # Initialize Q-table
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.random.uniform(low=-0.1, high=0.1, size=(n_states, n_actions))
    q_table[(env.desc == b"G").flatten()] = 0  # Assuming 'G' is the goal/terminal state

    epsilon = initial_epsilon
    rewards, lengths = [], []

    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        truncated = False
        total_reward, steps = 0, 0

        while not (done or truncated):
            # Epsilon-greedy action selection
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state, :])  # Exploit learned values

            next_state, reward, done, truncated, info = env.step(action)

            # Q-Learning update rule
            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
            )
            total_reward += reward

            state = next_state

            steps += 1

        # Epsilon decay
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Evaluation
        if (episode + 1) % eval_every == 0:
            avg_reward, avg_length = evaluate_policy(env, q_table, eval_episodes)
            rewards.append(avg_reward)
            lengths.append(avg_length)
            # print(f"Episode: {episode + 1}, Avg. Reward: {avg_reward}, Avg. Length: {avg_length}, Epsilon: {epsilon}")

    print("Training completed.")
    return q_table, rewards, lengths


def train_double_q_learning(
    env: gym.Env,
    alpha: float = 0.1,
    gamma: float = 0.99,
    initial_epsilon: float = 1.0,
    min_epsilon: float = 0.1,
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
    q_table_a = np.random.uniform(low=0, high=0.1, size=(n_states, n_actions))
    q_table_b = np.random.uniform(low=0, high=0.1, size=(n_states, n_actions))
    q_table_a[
        (env.desc == b"G").flatten()
    ] = 0  # Assuming 'G' is the goal/terminal state
    q_table_b[
        (env.desc == b"G").flatten()
    ] = 0  # Assuming 'G' is the goal/terminal state

    epsilon = initial_epsilon
    rewards, lengths = [], []

    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        truncated = False
        total_reward, steps = 0, 0

        while not (done or truncated):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table_a[state, :] + q_table_b[state, :])

            next_state, reward, done, truncated, info = env.step(action)
            if np.random.rand() < 0.5:
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

            total_reward += reward
            steps += 1
            state = next_state

        epsilon = max(min_epsilon, epsilon_decay * epsilon)

        if (episode + 1) % eval_every == 0:
            avg_reward, avg_length = evaluate_policy(
                env, q_table_a + q_table_b, eval_episodes
            )
            rewards.append(avg_reward)
            lengths.append(avg_length)

    print("Double Q-learning training completed.")
    return q_table_a, q_table_b, rewards, lengths


def train_speedy_q_learning(
    env: gym.Env,
    alpha: float = 0.1,
    gamma: float = 0.99,
    initial_epsilon: float = 1.0,
    min_epsilon: float = 0.1,
    epsilon_decay: float = 0.995,
    episodes: int = 10000,
    eval_every: int = 100,
    eval_episodes: int = 10,
) -> np.ndarray:
    """
    Trains an agent using the Speedy Q-Learning algorithm on a specified environment.

    Args:
        env (gym.Env): The environment to train the agent on.
        alpha (float): Learning rate.
        gamma (float): Discount factor for future rewards.
        episodes (int): Total number of training episodes.

    Returns:
        np.ndarray: The final Q-table learned by the agent.
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    # q_table = np.zeros((n_states, n_actions))
    # q_table_old = np.zeros((n_states, n_actions))
    q_table = np.random.uniform(low=0, high=0.1, size=(n_states, n_actions))
    q_table_old = np.copy(q_table)  # To store the Q-table from the previous iteration

    epsilon = initial_epsilon
    rewards, lengths = [], []
    total_reward, steps = 0, 0

    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        truncated = False

        while not (done or truncated):
            if np.random.rand() < epsilon:
                action = np.random.choice(n_actions)
            action = np.argmax(q_table[state])
            next_state, reward, done, truncated, info = env.step(action)

            # Speedy Q-Learning Update Rule
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state][best_next_action] * (
                not done
            )
            td_delta = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_delta

            # Speedy Q-learning update
            beta = 0.5 * alpha
            speedy_td_target = reward + gamma * q_table_old[next_state][
                best_next_action
            ] * (not done)
            speedy_td_delta = speedy_td_target - q_table_old[state][action]
            q_table[state][action] += beta * (td_delta + speedy_td_delta)

            # Store current Q-table as previous Q-table for next iteration
            q_table_old = np.copy(q_table)

            total_reward += reward
            steps += 1

            state = next_state

        epsilon = max(min_epsilon, epsilon_decay * epsilon)

        if (episode + 1) % eval_every == 0:
            avg_reward, avg_length = evaluate_policy(env, q_table, eval_episodes)
            rewards.append(avg_reward)
            lengths.append(avg_length)

    return q_table, rewards, lengths


def evaluate_policy(env, q_table, episodes=10):
    """Evaluate the Q-learning agent for a certain number of episodes and return average reward and steps."""
    total_reward, total_length = 0, 0

    for _ in range(episodes):
        state = env.reset()[0]
        done = False
        episode_reward, steps = 0, 0

        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

        total_reward += episode_reward
        total_length += steps

    avg_reward = total_reward / episodes
    avg_length = total_length / episodes
    return avg_reward, avg_length


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


def visualize_frozen_lake_double(q_table_a, q_table_b, env_name="FrozenLake-v1"):
    env = gym.make(env_name, render_mode="human")
    state = env.reset()[0]
    env.render()
    done = False
    total_reward = 0
    while not done:
        action = np.argmax(
            (q_table_a[state] + q_table_b[state]) / 2
        )  # Use the average of both Q-tables
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        print(f"Action: {action}, State: {state}, Reward: {reward}")
        if done:
            print(f"Episode finished with a total reward of: {total_reward}")


if __name__ == "__main__":
    # Create the FrozenLake environment
    env = gym.make("FrozenLake-v1")

    # Train the agent using Standard Q learning
    # q_table, rewards, lengths = train_q_learing(env)
    # _plot_evaluation(rewards, lengths)
    # visualize_frozen_lake(q_table)

    # Train the agent using Double Q learning
    # q_table_a, q_table_b, rewards, lengths = train_double_q_learning(env)
    # _plot_evaluation(rewards, lengths)
    # visualize_frozen_lake_double(q_table_a, q_table_b)
    # print("Trained Q-Table:")
    # print(q_table_a)
    # print("Trained Q-Table:")
    # print(q_table_b)

    # Train the agent using Speedy Q Learning
    q_table, rewards, lengths = train_speedy_q_learning(env)
    _plot_evaluation(rewards, lengths)
    visualize_frozen_lake(q_table)
    print("Trained Q-Table:")
    print(q_table)

    # sql = SpeedyQLearning(env.observation_space.n, env.action_space.n)
    # sql.train(env)
    # q_table = sql.q_table
    # # _plot_evaluation(rewards, lengths)
    # visualize_frozen_lake(sql.q_table)
