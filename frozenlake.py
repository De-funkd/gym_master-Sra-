import gym
import numpy as np

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human")

gamma = 0.9
alpha = 0.1
epsilon = 0.1
episodes = 100
decay = 0.995

states = env.observation_space.n
actions = env.action_space.n
Q = np.zeros((states, actions))


def choose_action(state):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])


def training():
    global epsilon
    for episode in range(episodes):
        state, _ =env.reset()
        done=False
        total = 0

        while not done:
            action = choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            total += reward


            best_action = np.argmax(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * Q[next_state, best_action] - Q[state, action])

            state = next_state


        epsilon=epsilon*decay



    print("Training complete.")


def run():
    state, _ = env.reset()
    env.render()

    steps = 0
    done = False

    while not done:
        action = np.argmax(Q[state])
        state, reward, done, _, _ = env.step(action)
        env.render()
        steps += 1

        if done:
            if reward == 1:
                print("goal achieved")
            else:
                print("oops sorry you failed")
            break


q_learning()

run()
