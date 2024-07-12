import numpy as np
import matplotlib.pyplot as plt

# Parameterzzz
k = 100
steps = 100
epsilons = [0.01, 0.1, 0]

# initializing true rewards (generating randomly)
true_rewards = np.random.randn(k)

#choosing actions acc. to the epsilon vale 
def choose_action(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(k) 
    else:
        return np.argmax(q_values)  

# gives reward for the action (it is random) , since rewards will be random IRL
def get_reward(action):
    return np.random.randn() + true_rewards[action]

# updatinng action-values 
def update_estimates(q_values, action_counts, action, reward):
    action_counts[action] += 1
    q_values[action] += (reward - q_values[action]) / action_counts[action]

# runss the bandit and collects avg. reward
def run_bandit(epsilon, steps):
    q_values = np.zeros(k)
    action_counts = np.zeros(k)
    rewards = np.zeros(steps)
    avg_rewards = np.zeros(steps)
    
    for tstep in range(steps):
        action = choose_action(q_values, epsilon)
        reward = get_reward(action)
        update_estimates(q_values, action_counts, action, reward)
        rewards[tstep] = reward
        avg_rewards[tstep] = np.mean(rewards[:tstep+1])
        
            
    return avg_rewards

#iterates thr and runs the code for all values of epsilon 
avg_rewards_epsilons = {}
for epsilon in epsilons:
    avg_rewards_epsilons[epsilon] = run_bandit(epsilon, steps)

# plots avg rewards v/s epsilon value curve 

plt.figure(figsize=(10, 6))
for epsilon in epsilons:
    plt.plot(avg_rewards_epsilons[epsilon], label=f'epsilon = {epsilon}')
plt.xlabel('Time Steps')
plt.ylabel('Average Reward')
plt.title('Average Reward vs. Time Steps for Different Epsilon Values')
plt.legend()
plt.grid(True)
plt.show()

#prints avg rewards for each value of epsilon 
for epsilon in epsilons:
    print(f"Average reward over {steps} steps for epsilon {epsilon}: {avg_rewards_epsilons[epsilon][-1]}")
