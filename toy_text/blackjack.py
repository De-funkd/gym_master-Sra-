import gymnasium as gym
import numpy as np

env = gym.make("Blackjack-v1")

# Parameters
gamma = 1.0
numberofepisodes = 10000  
epsilon = 1.0  
epsilon_decay = 0.999 
min_epsilon = 0.01 

actions = env.action_space.n
Q = {}

def choose_action(state):
    if state not in Q:
        Q[state] = np.zeros(actions)
    if np.random.random() < epsilon:
        return env.action_space.sample() 
    else:
        return np.argmax(Q[state])

def generate_episode(env):
    episode = []
    state, _ = env.reset()
    done = False

    while not done:
        action = choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
    
    return episode

# monte carlo 
def monte_carlo_control():
    returns_sum = {}
    returns_count = {}
    global epsilon

    for numberofepisode in range(numberofepisodes):
        episode = generate_episode(env)
        G = 0
        visited_state_action_pairs = set()

        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            if (state, action) not in visited_state_action_pairs:
                visited_state_action_pairs.add((state, action))
                if (state, action) not in Q:
                    Q[state] = np.zeros(actions)
                if (state, action) not in returns_sum:
                    returns_sum[(state, action)] = 0.0
                    returns_count[(state, action)] = 0.0
                returns_sum[(state, action)] += G
                returns_count[(state, action)] += 1.0
                Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]

        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    print("Training complete.")

# Train the model using Monte Carlo 
monte_carlo_control()

def test(env):
    for _ in range(20):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.argmax(Q.get(state, np.zeros(actions)))
            next_state, reward, done, _, _ = env.step(action)

            if done:
                print(f"Final State: {state}, Reward: {reward}")
                break
            state = next_state

test_env = gym.make("Blackjack-v1", render_mode="human")
test(test_env)

env.close()
test_env.close()
