import gym
import numpy as np

# Parameters
episodes = 10000  
epsilonn = 0.1  

env = gym.make('Blackjack-v1', render_mode='rgb_array', sab=True, new_step_api=True)

Q_table = np.zeros((18, 10, 2))

def policy(playersum, dealercard, epsilon=epsilonn):
    if np.random.rand() < epsilon:
        return np.random.choice([0, 1])  
    else:
        return np.argmax(Q_table[playersum, dealercard]) 

# Monte Carlo Loop
for episode in range(episodes):  

    state = env.reset()
    
    if isinstance(state, tuple):
        state = state[0]
    
    episode_data = []
    
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        
        if isinstance(state, tuple):
            player_sum, dealer_card, _ = state
        else:
            player_sum, dealer_card, _ = (state, state, False)

        # Storing the values according to the indexing in the Q-table 
        playersum = player_sum - 4
        dealercard = dealer_card - 1
        
        dealercard = min(max(dealercard, 0), 9)
        
        action = policy(playersum, dealercard)
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        episode_data.append((playersum, dealercard, action, reward))
        
        state = next_state
    
    # Averaging the reward:
    
    G = 0
    for playersum, dealercard, action, reward in reversed(episode_data):
        G += reward
        Q_table[playersum, dealercard, action] += (G - Q_table[playersum, dealercard, action]) / (episode + 1)

# Save the Q-table 
np.save('qtable_blackjack.npy', Q_table)

Q_table_loaded = np.load('qtable_blackjack.npy')

# Running the trained model:

env = gym.make('Blackjack-v1', render_mode='human', sab=True, new_step_api=True)  
trainingepisodes = int(10)

for episode in range(trainingepisodes):  
    state = env.reset()
    
    terminated, truncated = False, False
    
    while not (terminated or truncated):
        
        if isinstance(state, tuple):
            player_sum, dealer_card, _ = state
        else:
            player_sum, dealer_card, _ = (state, state, False)
        
        playersum = player_sum - 4
        dealercard = dealer_card - 1
        
        dealercard = min(max(dealercard, 0), 9)
        
        action = np.argmax(Q_table_loaded[playersum, dealercard])
        
        state, reward, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            if reward > 0:
                result = "Win"
            elif reward < 0:
                result = "Loss"
            else:
                result = "Draw"
            
            print(f"Player's card sum is: {player_sum}, Dealer's card is: {dealer_card + 1}, Result is: {result}")

