import numpy as np
import math

n1 = input("enter the co-ordinate for the win terminal cell(0-3)")
m1 = input("enter the co-ordinate for the win terminal cell(0-3)")

n1, m1 = int(n1) , int(m1)


n2 = input("enter the co-ordinate for the loose terminal cell(0-3)")
m2 = input("enter the co-ordinate for the loose terminal cell(0-3)")

n2, m2 = int(n2), int(m2)

value = np.zeros((4,4))
policy = np.full((4, 4), '', dtype=object)
print("Values of matrix initially are", value)
gamma = 0.9
prob = 0.8
notprob = 0.2 / 3
actions = ['up', 'down', 'left', 'right']
rows = value.shape[0]
cols = value.shape[1]

win = [(n1,m1)]
lose = [(n2,m2)]


def control(i, j, action):
    if action == 'up':
        if i > 0:
            new_i = i - 1
        else:
            new_i = 0
        new_j = j
    elif action == 'down':
        if i < rows - 1:
            new_i = i + 1
        else:
            new_i = rows - 1
        new_j = j
    elif action == 'right':
        new_i = i
        if j < cols - 1:
            new_j = j + 1
        else:
            new_j = cols - 1
    elif action == 'left':
        new_i = i
        if j > 0:
            new_j = j - 1
        else:
            new_j = 0
    else:
        new_i = i
        new_j = j
    return new_i, new_j


def compute():
    global value, policy
    theta = math.pow(10, -6)
    maxsteps = 100

    for step in range(maxsteps):
        new_value = np.copy(value)
        change = 0

        for i in range(rows):
            for j in range(cols):

                if (i, j) in win:
                    reward1 = 1
                elif (i, j) in lose:
                    reward1 = -1
                else:
                    reward1 = 0

                max_value = -1000000
                best_action = ''

                for action in actions:

                    new_i, new_j = control(i, j, action)

                    if (new_i, new_j) in win:
                        next_reward = 1
                    elif (new_i, new_j) in lose:
                        next_reward = -1
                    else:
                        next_reward = 0

                    intended_value = prob * (reward1 + next_reward + gamma * value[new_i, new_j])

                    unintended_value = 0 
                    for noaction in actions:
                        if noaction != action:
                            next_i_unintended, next_j_unintended = control(i, j, noaction)

                            if (next_i_unintended, next_j_unintended) in win:
                                unintended_next_reward = 1
                            elif (next_i_unintended, next_j_unintended) in lose:
                                unintended_next_reward = -1
                            else:
                                unintended_next_reward = 0

                            unintended_value += notprob * (
                                reward1 + unintended_next_reward + gamma * value[next_i_unintended, next_j_unintended]
                            )

                    actionval = intended_value + unintended_value
                    if actionval > max_value:
                        best_action = action


                    max_value = max(max_value, actionval)
                    policy[i,j] = best_action


                new_value[i, j] = max_value
                 
                
                change = max(change, abs(value[i, j] - new_value[i, j]))

        value = new_value

        for i in range(rows) :
                        for j in range(rows) :
                            
                            if value[i,j] == np.max(value) :

                                policy[i,j] = 0 

        if change < theta:
            break

    print("Updated value function:\n", value)
    print("Optimal policy:\n", policy)

run = compute()
