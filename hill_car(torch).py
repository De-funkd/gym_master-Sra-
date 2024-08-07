import torch
import gym
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = gym.make("MountainCar-v0", render_mode="human")
episodes = 2500
steps = 1000

obs_pos = torch.linspace(env.observation_space.low[0], env.observation_space.high[0], 40).to(device)
obs_vel = torch.linspace(env.observation_space.low[1], env.observation_space.high[1], 40).to(device)
q_table = torch.zeros((40, 40, 3), device=device)

alpha = 0.9
gamma = 0.8
epsilon = 0.1

def take_action(state_pos, state_vel, epsilon):
    if torch.rand(1).item() < epsilon:
        return env.action_space.sample()
    else:
        return torch.argmax(q_table[state_pos, state_vel, :]).item()

for i in range(episodes):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    state_pos = torch.searchsorted(obs_pos, torch.tensor(state[0], device=device)).item() - 1
    state_vel = torch.searchsorted(obs_vel, torch.tensor(state[1], device=device)).item() - 1
    total = 0

    for step in range(steps):
        action = take_action(state_pos, state_vel, epsilon)
        step_result = env.step(action)

        if len(step_result) == 4:
            next_obs, reward, done, _ = step_result
            terminate = truncated = done
        else:
            next_obs, reward, terminate, truncated, _ = step_result

        next_pos = torch.searchsorted(obs_pos, torch.tensor(next_obs[0], device=device)).item() - 1
        next_vel = torch.searchsorted(obs_vel, torch.tensor(next_obs[1], device=device)).item() - 1

        reward = 100 if next_obs[0] >= 0.6 else -1

        next_action = torch.argmax(q_table[next_pos, next_vel, :]).item()
        target = reward + gamma * q_table[next_pos, next_vel, next_action]
        q_table[state_pos, state_vel, action] += alpha * (target - q_table[state_pos, state_vel, action])

        state_pos, state_vel = next_pos, next_vel
        total += reward

        if terminate or truncated:
            break

    print(f"Episode {i + 1} of {episodes} completed - Total Reward: {total}")

torch.save(q_table, "q_table.pt")
env.close()

# Training ends here and now we begin running the trained model
q_table = torch.load("q_table.pt")
env = gym.make("MountainCar-v0", render_mode="human")
new_episodes = 10
new_steps = 1000

for i in range(new_episodes):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    state_pos = torch.searchsorted(obs_pos, torch.tensor(state[0], device=device)).item() - 1
    state_vel = torch.searchsorted(obs_vel, torch.tensor(state[1], device=device)).item() - 1
    total = 0

    for step in range(new_steps):
        action = torch.argmax(q_table[state_pos, state_vel, :]).item()
        step_result = env.step(action)

        if len(step_result) == 4:
            next_obs, reward, done, _ = step_result
            terminate = truncated = done
        else:
            next_obs, reward, terminate, truncated, _ = step_result

        next_pos = torch.searchsorted(obs_pos, torch.tensor(next_obs[0], device=device)).item() - 1
        next_vel = torch.searchsorted(obs_vel, torch.tensor(next_obs[1], device=device)).item() - 1

        state_pos, state_vel = next_pos, next_vel
        total += reward

        if terminate or truncated:
            break

    print(f"Test Episode {i + 1} of {new_episodes} completed - Total Reward: {total}")

