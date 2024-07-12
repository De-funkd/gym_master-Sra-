import numpy as np
import matplotlib.pyplot as plt

k = 500 # denotes total number of slot machines
t = 500 # denotes the total timesteps i.e how many times arms will be pulled
eval = [0,0.1, 0.05, 0.01] # denotes list of different epsilon values

mean = np.random.rand(k) # generates true mean reward for each arm randomly between 0 and 1
rewards = np.zeros((len(eval), t)) #2d array to store rewards for each epsilon value and timestep
avg = np.zeros((len(eval), t)) # stores the average reward uptill a particular time step for different epsilon values
best_arm_counts = np.zeros(len(eval))  #1D array that stores the number of times a best arm encountered for particular epsilon value

def select(q_value, epsilon): 
    if np.random.rand() < epsilon:
        return np.random.randint(0, k) #randomly selects an arm favouring exploration
    else:
        return np.argmax(q_value)# selects arm with maximum mean reward favouring exploitation

def pullarm(arm): # chooses a particular arm 
    return mean[arm] #chooses true mean reward for arm thats chosen

def update(q_value, N, arm, reward): #updates number of times a arm is played and the value of estimated mean reward of that arm
    N[arm] += 1
    q_value[arm] += (reward - q_value[arm]) / N[arm]

best_arm = np.argmax(mean) 

for i in range(len(eval)): 
    epsilon = eval[i] # selects particular epsilon value
    q_value = np.zeros(k) 
    N = np.zeros(k)

    for step in range(t):
        arm = select(q_value, epsilon) # for a time step uptill max time step we select arm
        reward = pullarm(arm)# then we pull an arm
        rewards[i, step] = reward # store the reward for particular arm at particular time step
        avg[i, step] = np.mean(rewards[i, :step + 1]) #store average reward till a particular time step
        update(Q, N, arm, reward) #update estimated mean reward for chosen arm

        if arm == best_arm: 
            best_arm_counts[i] += 1 # increment best arm count if arm chosen yields max reward

time = np.arange(t) 
avgrewards = np.mean(rewards, axis=1) #calcultes and stores mean reward for each arm

plt.figure(figsize=(10, 5)) #used to setup the canvas for plotting
for i in range(len(eval)): 
    plt.plot(time, avg[i], label=f"Epsilon = {eval[i]}") #plots a graph for average of each epsilon value vs time and labels it
 
plt.xlabel("Time Steps") 
plt.ylabel("Average Reward")
plt.title("Average Reward vs Time Steps for Different Epsilon Values")
plt.legend()
plt.show()


probabilities = best_arm_counts / t #calculates probability of choosing best arm

for i in range(len(eval)):
    print(f"For epsilon: {eval[i]}") 
    print("Average reward:", avgrewards[i])
    print("Probability of selecting the best arm:", probabilities[i])
    print("\n")
