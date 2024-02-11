import numpy as np
import gymnasium as gym


# Create the FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=True)


def train_frozen_lake(env, alpha=0.1, gamma=0.99, epsilon=0.1, episodes=10000):
    # Initialize Q-table
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    q_table = np.zeros((n_states, n_actions))

    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        truncated = False

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

            state = next_state

    print("Training completed.")
    return q_table


# Train the agent
q_table = train_frozen_lake(env)

# Display the trained Q-table
print("Trained Q-Table:")
print(q_table)


def visualize_frozen_lake(q_table, env_name="FrozenLake-v1"):
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


visualize_frozen_lake(q_table)
