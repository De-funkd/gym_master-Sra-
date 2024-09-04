import numpy as np
import matplotlib.pyplot as plt

k = 100  # number of actions
steps = 1000  # number of steps to simulate
epsilons = [0.01, 0.1, 0]  # different epsilon values

# Generate rewards for arm
true_rewards = np.random.randn(k)

#  storage for average rewards
avg_rewards_epsilons = {}

for ep in epsilons:
    q_val = np.zeros(k)
    N = np.zeros(k)
    rewards = np.zeros(steps)
    avg_rewards = np.zeros(steps)
    
    total_reward = 0
    
    for t in range(steps):
        if np.random.rand() < ep:
            action = np.random.randint(k)  # exploration
        else:
            action = np.argmax(q_val)  # exploitation
        
        reward =  true_rewards[action]
        N[action] += 1
        q_val[action] += (reward - q_val[action]) / N[action]
        
        rewards[t] = reward
        total_reward += reward
        avg_rewards[t] = total_reward / (t + 1)
    
    avg_rewards_epsilons[ep] = avg_rewards

# Plot the results
plt.figure(figsize=(10, 6))
for epsilon in epsilons:
    plt.plot(avg_rewards_epsilons[epsilon], label=f'epsilon = {epsilon}')
plt.xlabel('Time Steps')
plt.ylabel('Average Reward')
plt.title('Average Reward vs. Time Steps for Different Epsilon Values')
plt.legend()
plt.grid(True)
plt.show()

# Print average rewards at the end
for epsilon in epsilons:
     print('Epsilon', epsilon, ': Avg reward =', round(avg_rewards_epsilons[epsilon][-1], 2))

