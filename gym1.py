import numpy as np
import matplotlib.pyplot as plt

k = 1000
t = 1000
eval = [0.1, 0.05, 0.01]

mean = np.random.rand(k)
rewards = np.zeros((len(eval), t))
avg = np.zeros((len(eval), t))
best_arm_counts = np.zeros(len(eval))  

def select(Q, e):
    if np.random.rand() < e:
        return np.random.randint(0, k)
    else:
        return np.argmax(Q)

def pullarm(arm):
    return mean[arm]

def update(Q, N, arm, reward):
    N[arm] += 1
    Q[arm] += (reward - Q[arm]) / N[arm]

best_arm = np.argmax(mean)

for i in range(len(eval)):
    e = eval[i]
    Q = np.zeros(k)
    N = np.zeros(k)

    for step in range(t):
        arm = select(Q, e)
        reward = pullarm(arm)
        rewards[i, step] = reward
        avg[i, step] = np.mean(rewards[i, :step + 1])
        update(Q, N, arm, reward)

        if arm == best_arm:
            best_arm_counts[i] += 1

time = np.arange(t)
avgrewards = np.mean(rewards, axis=1)

plt.figure(figsize=(10, 5))
for i in range(len(eval)):
    plt.plot(time, avg[i], label=f"Epsilon = {eval[i]}")

plt.xlabel("Time Steps")
plt.ylabel("Average Reward")
plt.title("Average Reward vs Time Steps for Different Epsilon Values")
plt.legend()
plt.show()


probabilities = best_arm_counts / t

for i in range(len(eval)):
    print(f"For epsilon: {eval[i]}")
    print("Average reward:", avgrewards[i])
    print("Probability of selecting the best arm:", probabilities[i])
    print("\n")
