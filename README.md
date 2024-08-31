## REINFORCEMENT LEARNING

![image](https://github.com/user-attachments/assets/fb57cc63-4175-48fe-8404-4797297112c5)


## Aim:
1) Understanding concepts of reinforcement learning
2) Solving basic implementations like k armed bandits and gridworld
3) To solve implementations on open ai gym
4) implementing atari game of pong on gym


## Table of Contents
| Sr No | Content                        | Pg No |
|-------|--------------------------------|-------|
| 1     | Overview                       |       |
| 2     | Introduction:                  |       |
|       | 1) What is RL?                 |       |
|       | 2) What is Q-learning?         |       |
| 3     | Implementations:               |       |
|       | 1) Mountain Car Implementation |       |
|       | 2) Cartpole Implementation     |       |
|       | 3) Blackjack Implementation    |       |
|       | 4) Frozen Lake Implementation  |       |
| 4     | Deep Q-Learning:               |       |
|       | 1) What is DQN?                |       |
|       | 2) Pong Implementation         |       |
|       | 3) Commands Required           |       |
| 5     | Future Works                   |       |
| 6     | References                     |       |
   

## Theory:
Refer our documentation for getting a deeper insight into our project

Platforms Used:

## Open AI GYM


![image](https://github.com/user-attachments/assets/0ff8bd67-c583-44a6-9361-fcdfa0677bef)


## Kaggle

![image](https://github.com/user-attachments/assets/edf2db32-6e79-43ca-bd96-4c9551b62406)





File Structure


![image](https://github.com/user-attachments/assets/db53041f-57c1-4c57-94a4-59b60239e7c3)




## What is RL?

  Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment to achieve a goal. 
  The agent receives feedback in the form of rewards or penalties based on its actions, which it uses to learn optimal strategies or policies.
  It consists of actions, states, rewards and environment


![image](https://github.com/user-attachments/assets/0f0aa061-a6a9-4b6a-8974-7b3cc5afb02f)





## Q learning
Q-learning is an off-policy algorithm that aims to learn the value of state-action pairs to determine the best policy for maximizing cumulative rewards in each environment. It updates the Q-values (expected future rewards) based on the agent's experiences to improve decision-making.

Implementations

 ## 1) Mountain Car


   https://github.com/user-attachments/assets/ae525c3b-e0e7-4c33-9df5-4189ee670474
   
    
 

After training for 2500 episodes we observe the car reaching the flagpost

## 2)Cartpole


   
https://github.com/user-attachments/assets/9f79fa41-6ff4-4750-9be5-ac78f052078e



After training for 2000 episodes we observe the pole being balanced on the cart successfully

3) ## 3) Blackjack
      Implementing monte carlo learning to get desired results

           ![image](https://github.com/user-attachments/assets/550aa41c-fbd8-43f0-b01c-7355c5015c20)



5) ## 4) Frozen lake

    Implementing frozen lake problem and achieving results after 1000 episodes training



 
   
   ![frozenlake](https://tse4.mm.bing.net/th?id=OIP.TntNAltZ4iO0puJRIUSQRgAAAA&pid=Api&P=0&h=180)





Process to train and run the code:
 * First we train the code on a platform called kaggle from where we recieve a q table of values
*  It gets saved as a .npy file
*  We copy paste this file into our terminal where we run the code based on this q table
*  We make basic installations for gym,numpy and pygame


## PONG
Deep Q networks:
 
[Pong](https://mir-s3-cdn-cf.behance.net/project_modules/max_1200/c08edd97535089.5ec71d61c627a.gif)
 



Deep Q-Networks (DQN) is an advanced reinforcement learning algorithm that combines Q-learning with deep learning to handle complex environments with large or continuous state spaces. 

We implement pong using convolutional neural networks. 
1.	Initialize Environment: Use OpenAI Gym's Pong environment.
2.	Build CNN Model: Create a neural network with convolutional layers to process image frames and predict Q-values.
3.	Train with DQN: Use Deep Q-Learning, which combines CNNs for feature extraction with Q-learning for decision-making. Incorporate experience replay and a target network for stability.
4.	Action Selection: Use the CNN to choose actions based on predicted Q-values and update the model based on rewards.


![image](https://github.com/user-attachments/assets/26e68d1b-4575-431a-b46e-8816aca7d8d1)



Implementation of Pong



![image](https://github.com/user-attachments/assets/89d9ee5f-932b-406b-ae4d-12efbf58f2ea)




Installations required for Pong
Gym
Numpy
Cv
Pygame

![image](https://github.com/user-attachments/assets/0dc5908c-03a4-4953-8727-11b844653d4a)


![image](https://github.com/user-attachments/assets/65309a51-e436-4026-b7f4-4d9c950f345f)

![image](https://github.com/user-attachments/assets/28aadf4d-d530-4412-9e9d-fe6425b3231f)




Future goals:
*	Implementing pong using libraries such as keras and TensorFlow for better and efficient results
*	Implementation of RL on a self-balancing bot
* Usage in simulation of bots on platforms like gazebo
*	Developing a RL based drone to deliver packages

Troubleshooting
* Make sure to use the correct name of the environment
* Use the updated version of gym when running the pong code

   
