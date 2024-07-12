import numpy as np
import matplotlib.pyplot as plt

def k_armed_bandit(k, t, epsilon=0.1):

    # Set up the real average rewards for each arm (action)
    mean = np.random.rand(k)
    # Start with no knowledge about any arm's value
    q_value = np.zeros(k)
    # Keep track of how many times each arm has been pulled
    N = np.zeros(k)
    # List to store the reward from each step
    rewards = []
    # List to store the average reward over time
    average_rewards = []

    def select_action(q_value, N, epsilon):
        # Decide if we should explore (try a random arm) or exploit (choose the best-known arm)
        if np.random.rand() < epsilon:
            return np.random.randint(0, k)  # Try a random arm
        else:
            return np.argmax(q_value)  # Pick the arm that seems the best

    def pull_arm(arm):
        # Simulate pulling the chosen arm and getting a reward
        return np.random.normal(mean[arm], 1.0)

    def update(q_value, N, arm, reward):
        # Update our knowledge about the chosen arm based on the new reward
        N[arm] += 1
        q_value[arm] += (reward - q_value[arm]) / N[arm]

    # Run the bandit algorithm for the given number of steps
    for step in range(t):
        action = select_action(q_value, N, epsilon)  # Choose an arm
        reward = pull_arm(action)  # Get a reward
        update(q_value, N, action, reward)  # Update our knowledge
        rewards.append(reward)  # Save the reward
        average_rewards.append(np.mean(rewards))  # Update the average reward

    return rewards, average_rewards

def plot_average_reward_vs_time(average_rewards, title):
    plt.plot(average_rewards)
    plt.xlabel('Time-step')
    plt.ylabel('Average Reward')
    plt.title(title)
    plt.show()

# Parameters
k = 5  # Number of arms
t = 1000  # Number of time steps
epsilon_values = [0.1, 0.2, 0.5]  # Different exploration rates to try

# Run the k-armed bandit for different epsilon values and plot the results
for epsilon in epsilon_values:
    rewards_eps, average_rewards_eps = k_armed_bandit(k, t, epsilon)
    plot_average_reward_vs_time(average_rewards_eps, f'Epsilon-Greedy (epsilon={epsilon}): Average Reward vs Time-step')
