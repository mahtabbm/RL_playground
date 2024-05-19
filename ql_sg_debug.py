import numpy as np
from typing import List, Tuple
import gym_simplegrid
from datetime import datetime as dt
import gymnasium as gym
import matplotlib.pyplot as plt

def evaluate_policy(env, q_table, episodes=10):
    """Evaluate the Q-learning agent for a certain number of episodes and return average reward and steps."""
    total_reward, total_length = 0, 0
    # options ={
    #     'start_loc': 0,
    #     'goal_loc': 15
    # }

    for _ in range(episodes):

        state = env.reset(seed=1234, options={'start_loc':0, 'goal_loc':63})[0]
        done = truncated = False
        episode_reward, steps = 0, 0

        while not (done or truncated):
            action = np.argmax(q_table[state])

            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            # if steps >200:
            #     episode_reward = 0
            #     print("eval break")
            #     break

        total_reward += episode_reward
        total_length += steps

    avg_reward = total_reward / episodes
    avg_length = total_length / episodes
    return avg_reward, avg_length


def _plot_evaluation(rewards: List[float], lengths: List[int], title):
    # Plotting
    plt.figure(figsize=(12, 5))

    # Plot for average cumulative rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, 'o-')  # Add 'o' marker
    plt.title("Average Cumulative Reward vs. Evaluation Episodes")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Average Cumulative Reward")

    # Plot for average steps
    plt.subplot(1, 2, 2)
    plt.plot(lengths, 'o-')  # Add 'o' marker
    plt.title("Average Steps vs. Evaluation Episodes")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Average Steps")
    
    plt.tight_layout()
    plt.suptitle(title)  # Adds a title to the entire figure
    plt.show()

def universal_initialize_q_table(env):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))
    return q_table


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
    # n_states = env.observation_space.n
    # n_actions = env.action_space.n
    # q_table = np.random.uniform(low=-0.1, high=0.1, size=(n_states, n_actions))
    # q_table[(env.desc == b"G").flatten()] = 0  # Assuming 'G' is the goal/terminal state
    q_table = universal_initialize_q_table(env)
    # options ={
    #     'start_loc': 0,
    #     'goal_loc': 15
    # }

    env.reset(seed=1234, options={'start_loc':0, 'goal_loc':63})
    epsilon = initial_epsilon
    rewards, lengths = [], []
    first = True

    for episode in range(episodes):
        # print(episode)
        state = env.reset(seed=1234, options={'start_loc':0, 'goal_loc':63})[0]
        done = env.unwrapped.done
        total_reward, steps = 0, 0
        seq = []
        while not done:
            # Epsilon-greedy action selection
            if np.random.uniform(0, 1) <= epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state, :])  # Exploit learned values

            next_state, reward, done, truncated, info = env.step(action)

            # Q-Learning update rule
            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
            )
            total_reward += reward
            if reward == 1:
                print("reached the goal in state ", next_state)
                print("reward is ", reward, "total is ", total_reward)
                print("done is", done)
            seq.append(state)
            state = next_state
            steps += 1

        # Epsilon decay
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Evaluation
        print("The seq len is: ", len(seq))
        if (episode + 1) % eval_every == 0:
            avg_reward, avg_length = evaluate_policy(env, q_table, eval_episodes)
            if first and avg_reward:
                first = False
                print("The first episode reached to 1 is ", episode)
                print("The length is  ", avg_length)
            rewards.append(avg_reward)
            lengths.append(avg_length)
            print(f"Episode: {episode + 1}, Avg. Reward: {avg_reward}, Avg. Length: {avg_length}, Epsilon: {epsilon}")

    print("Training completed.")
    return q_table, rewards, lengths

if __name__ == '__main__':

    obstacle_map = [
            "10001000",
            "10010000",
            "00000001",
            "01000001",
        ]
    env = gym.make('SimpleGrid-8x8-v0', render_mode='rgb_array')

    q_table, rewards, lengths = q_learning(env, episodes=100, eval_episodes=1, eval_every=1)
    _plot_evaluation(rewards, lengths, title="Q-learning SG4x8")
    print(q_table)
