import numpy as np
import gymnasium as gym

env = gym.make("MountainCar-v0",render_mode = "human")

episodes = 1000
steps = 100

 
obs_pos = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 40)
obs_vel = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 40)

q_table = np.zeros((40, 40, 3)) 

alpha = 0.9
gamma = 0.8
epsilon = 0.1

def take_action(state_pos, state_vel, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state_pos, state_vel, :])

for i in range(episodes):
    state, info = env.reset()

    state_pos = np.digitize(state[0], obs_pos) - 1
    state_vel = np.digitize(state[1], obs_vel) - 1
    state = (state_pos, state_vel)

    total = 0

    for step in range(steps):
        action = take_action(state_pos, state_vel, epsilon)

        next_obs, reward, terminate, truncated, info = env.step(action)
        next_obs = np.array(next_obs)
        next_pos = np.digitize(next_obs[0], obs_pos) - 1
        next_vel = np.digitize(next_obs[1], obs_vel) - 1
        next_state = (next_pos, next_vel)

        if next_obs[0] >= 0.6:
            reward = 100
        else:
            reward = -1

        next_action = np.argmax(q_table[next_state])
        target = reward + gamma * q_table[next_state][next_action]
        q_table[state][action] += alpha * (target - q_table[state][action])

        state = next_state
        state_pos, state_vel = next_state

        total += reward
        if terminate or truncated:
            break

    
    print("Episode", i + 1, "of", episodes, "completed - Total Reward:", total)
np.save("q_table.npy", q_table)
env.close()

###training ends here and now we begin running the trained model 



q_table = np.load("q_table.npy")

env = gym.make("MountainCar-v0", render_mode="human")

new_episodes = 10  
new_steps = 1000

for i in range(new_episodes):
    state, info = env.reset()

    state_pos = np.digitize(state[0], obs_pos) - 1
    state_vel = np.digitize(state[1], obs_vel) - 1
    state = (state_pos, state_vel)

    total = 0

    for step in range(new_steps):
        action = np.argmax(q_table[state_pos, state_vel, :])

        next_obs, reward, terminate, truncated, info = env.step(action)
        next_obs = np.array(next_obs)
        next_pos = np.digitize(next_obs[0], obs_pos) - 1
        next_vel = np.digitize(next_obs[1], obs_vel) - 1
        next_state = (next_pos, next_vel)

        state = next_state
        state_pos, state_vel = next_state

        total += reward
        if terminate or truncated:
            break


