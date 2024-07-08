import numpy as np

k = 1000
e = 0.1
t = 10000

mean = np.random.rand(k)
Q = np.zeros(k)
N = np.zeros(k)
rewards = np.zeros(t)



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
best_arm_count = 0


for step in range(t):
    arm = select(Q, e)
    reward = pullarm(arm)
    rewards[step] = reward
    update(Q, N, arm, reward)


    if arm == best_arm:
      best_arm_count+= 1


avg = np.mean(rewards)


probability = best_arm_count / t


print("Average reward:", avg)
print("Probability of selecting the best arm:", probability)
