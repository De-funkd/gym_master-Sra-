import numpy as np
import gym

env = gym.make("CartPole-v0", render_mode="human")

episodes = 10000
steps = 500


obs_pos = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 40)
obs_vel = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 40)
obs_theta = np.linspace(env.observation_space.low[2], env.observation_space.high[2], 40)
obs_veltip = np.linspace(env.observation_space.low[3], env.observation_space.high[3], 40)


q_table = np.zeros((40, 40, 40, 40, 2))


alpha = 0.9
gamma = 0.8
epsilon = 0.1

def take_action(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

for i in range(episodes):
    state, info = env.reset()

    state_pos = np.digitize(state[0], obs_pos) - 1
    state_vel = np.digitize(state[1], obs_vel) - 1
    state_theta = np.digitize(state[2], obs_theta) - 1
    state_veltip = np.digitize(state[3], obs_veltip) - 1
    state = (state_pos, state_vel, state_theta, state_veltip)

    total = 0

    for step in range(steps):
        action = take_action(state, epsilon)

        next_obs, reward, terminate, truncated, info = env.step(action)

        next_pos = np.digitize(next_obs[0], obs_pos) - 1
        next_vel = np.digitize(next_obs[1], obs_vel) - 1
        next_theta = np.digitize(next_obs[2], obs_theta) - 1
        next_veltip = np.digitize(next_obs[3], obs_veltip) - 1
        next_state = (next_pos, next_vel, next_theta, next_veltip)


        if next_obs[0]>2.4:
            reward = -200
        else:
            reward = 1


        next_action = np.argmax(q_table[next_state])
        target = reward + gamma * q_table[next_state][next_action]
        q_table[state][action] += alpha * (target - q_table[state][action])

        state = next_state

        total += reward
        if terminate or truncated:
            break

    print(f"Episode {i + 1} of {episodes} completed - Total Reward: {total}")

np.save("q_table.npy", q_table)
env.close()


q_table = np.load("q_table.npy")

env = gym.make("CartPole-v0", render_mode="human")

new_episodes = 10
new_steps = 1000

for i in range(new_episodes):
    state, info = env.reset()

    state_pos = np.digitize(state[0], obs_pos) - 1
    state_vel = np.digitize(state[1], obs_vel) - 1
    state_theta = np.digitize(state[2], obs_theta) - 1
    state_veltip = np.digitize(state[3], obs_veltip) - 1
    state = (state_pos, state_vel, state_theta, state_veltip)

    total = 0

    for step in range(new_steps):
        action = np.argmax(q_table[state])

        next_obs, reward, terminate, truncated, info = env.step(action)

        next_pos = np.digitize(next_obs[0], obs_pos) - 1
        next_vel = np.digitize(next_obs[1], obs_vel) - 1
        next_theta = np.digitize(next_obs[2], obs_theta) - 1
        next_veltip = np.digitize(next_obs[3], obs_veltip) - 1
        next_state = (next_pos, next_vel, next_theta, next_veltip)

        state = next_state

        total += reward
        if terminate or truncated:
            break

    print(f"Test Episode {i + 1} completed - Total Reward: {total}")

env.close()
