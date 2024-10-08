# REINFORCEMENT LEARNING

![image](https://www.indianai.in/wp-content/uploads/2021/05/Reinforcement-Learning.jpg)



## Aim:
1) Understanding concepts of reinforcement learning
2) Solving basic implementations like k armed bandits and gridworld
3) To solve implementations on open ai gym
4) implementing atari game of pong on gym


## Table of Contents

| Sr No | Content                                    |
|-------|--------------------------------------------|
| 1     | [Aim](https://github.com/De-funkd/gym_master-Sra-/blob/main/README.md#aim)                          |
| 2     | [Introduction](https://github.com/De-funkd/gym_master-Sra-/blob/main/README.md#theory)                   |
|       | 2.1. [Theory](https://github.com/De-funkd/gym_master-Sra-/blob/main/README.md#theory)                    |
|       | 2.2. [Open Ai Gym](https://github.com/De-funkd/gym_master-Sra-/blob/main/README.md#Open-Ai-Gym)            |
|       | 2.3. [Kaggle](https://github.com/De-funkd/gym_master-Sra-/blob/main/README.md#Kaggle)            |
|       | 2.4. [File Structure](https://github.com/De-funkd/gym_master-Sra-/blob/main/README.md#File-Structure)               |
|       | 2.5. [What is Q-Learning?](https://github.com/De-funkd/gym_master-Sra-/blob/main/README.md#q-learning)       |
| 3     | [Implementations](https://github.com/De-funkd/gym_master-Sra-/blob/main/README.md#implementations)                |       
|       | 3.1. [Mountain Car Implementation](https://github.com/De-funkd/gym_master-Sra-/blob/main/README.md#1-mountain-car) |       
|       | 3.2. [Cartpole Implementation](https://github.com/De-funkd/gym_master-Sra-/blob/main/README.md#2-cartpole)   |       
|       | 3.3. [Blackjack Implementation](https://github.com/De-funkd/gym_master-Sra-/blob/main/README.md#3-blackjack)  |       
|       | 3.4. [Frozen Lake Implementation](https://github.com/De-funkd/gym_master-Sra-/blob/main/README.md#4-frozen-lake) |       
| 4     | [Deep Q-Learning](https://github.com/De-funkd/gym_master-Sra-/blob/main/README.md#deep-q-networks)                |       
|       | 4.1. [What is DQN?](https://github.com/De-funkd/gym_master-Sra-/blob/main/README.md#deep-q-networks)              |       
|       | 4.2. [Pong Implementation](https://github.com/De-funkd/gym_master-Sra-/blob/main/README.md#implementation-of-pong)       |       
|       | 4.3. [Commands Required](https://github.com/De-funkd/gym_master-Sra-/blob/main/README.md#installations-required-for-pong)         |       
| 5     | [Future Work](https://github.com/De-funkd/gym_master-Sra-/blob/main/README.md#future-goals)                    |       
| 6     | [References](https://github.com/De-funkd/gym_master-Sra-/blob/main/README.md#references-)                     |
| 7    | [Troubleshooting](https://github.com/De-funkd/gym_master-Sra-/blob/main/README.md#troubleshooting)                     |
  

   

## Theory:
Refer our  [documentation](https://1drv.ms/w/c/c682f7548892e17e/ESp9_4ueLFBKmo37eFAK4aABpeTRamopaGlbPDY7wfjmcg?e=J5fVEx) for getting a deeper insight into our project

## Platforms Used:

## Open AI GYM


![image](https://github.com/user-attachments/assets/0ff8bd67-c583-44a6-9361-fcdfa0677bef)


## Kaggle

![image](https://github.com/user-attachments/assets/edf2db32-6e79-43ca-bd96-4c9551b62406)





## File Structure


![image](https://github.com/user-attachments/assets/66befd0f-8fa5-4a73-b5d1-b95b7b8947f5)





## What is RL?

  Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment to achieve a goal. 
  The agent receives feedback in the form of rewards or penalties based on its actions, which it uses to learn optimal strategies or policies.
  It consists of actions, states, rewards and environment


![image](https://github.com/user-attachments/assets/0f0aa061-a6a9-4b6a-8974-7b3cc5afb02f)





## Q learning
Q-learning is an off-policy algorithm that aims to learn the value of state-action pairs to determine the best policy for maximizing cumulative rewards in each environment. It updates the Q-values (expected future rewards) based on the agent's experiences to improve decision-making.

## Implementations

 ## 1) Mountain Car


 

   ![Mountain Car](https://miro.medium.com/v2/resize:fit:1200/1*kn59uPbJKlD2spM1vVAbKg.gif)
   
    
 

After training for 2500 episodes we observe the car reaching the flagpost

## 2) Cartpole


   

![Cartpole](https://trencseni.com/images/cartpole.gif)



After training for 2000 episodes we observe the pole being balanced on the cart successfully


  ## 3) Blackjack
      
      

   ![Blackjack](https://www.gymlibrary.dev/_images/blackjack.gif)
  

   




   ## 4) Frozen lake

    Implementing frozen lake problem and achieving results after 1000 episodes training



 
   
   ![frozen lake](https://gymnasium.farama.org/_images/frozen_lake.gif)





Process to train and run the code:
 * First we train the code on a platform called kaggle from where we recieve a q table of values
*  It gets saved as a .npy file
*  We copy paste this file into our terminal where we run the code based on this q table
*  We make basic installations for gym,numpy and pygame


## PONG
 ## Deep Q networks:
 
[Pong](https://mir-s3-cdn-cf.behance.net/project_modules/max_1200/c08edd97535089.5ec71d61c627a.gif)
 



Deep Q-Networks (DQN) is an advanced reinforcement learning algorithm that combines Q-learning with deep learning to handle complex environments with large or continuous state spaces. 

We implement pong using convolutional neural networks. 
1.	Initialize Environment: Use OpenAI Gym's Pong environment.
2.	Build CNN Model: Create a neural network with convolutional layers to process image frames and predict Q-values.
3.	Train with DQN: Use Deep Q-Learning, which combines CNNs for feature extraction with Q-learning for decision-making. Incorporate experience replay and a target network for stability.
4.	Action Selection: Use the CNN to choose actions based on predicted Q-values and update the model based on rewards.


![image](https://github.com/user-attachments/assets/26e68d1b-4575-431a-b46e-8816aca7d8d1)



## Implementation of Pong




![pong](https://www.gymlibrary.dev/_images/pong.gif)




## Installations required for Pong
* [Gym](https://github.com/openai/gym)
* [Numpy](https://numpy.org/install/)
* [Cv](https://opencv.org/get-started/)
* [Pygame](https://www.pygame.org/download.shtml)
  

Use this [code repository](https://github.com/De-funkd/gym_master-Sra-)  and you're good to go!





 ## Future goals:
*	Implementing pong using libraries such as keras and TensorFlow for better and efficient results
*	Implementation of RL on a self-balancing bot
* Usage in simulation of bots on platforms like gazebo
*	Developing a RL based drone to deliver packages

## References :
 * Book - [Reinforcement Learning: An Introduction Richard S. Sutton and Andrew G. Barto](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
 * Video Reference - [Google Deepminds RL Playlist by David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
 * Courses - [Deep Learning Course 1 and 4 Basics of Neural Network and CNN by Andrew NG](https://www.coursera.org/specializations/machine-learning-introduction?adgroupid=161332312026&adposition=&campaignid=20594446971&creativeid=698085043093&device=c&devicemodel=&gad_source=1&gclid=CjwKCAjw_4S3BhAAEiwA_64YhsYARho42rZGRRY4A9L_R_FnOIX4RnPEqYJsHhOdrJII3BVsXoi5FBoC8S4QAvD_BwE&hide_mobile_promo&keyword=machine+learning+by+andrew+ng+coursera&matchtype=p&network=g&utm_campaign=b2c_india_machine-learning-introduction_stanford-deeplearning.ai_ftcof_specializations_arte_sep-23_dr_sem_rsa_gads_lg-all&utm_medium=sem&utm_source=gg)
* Research Paper - [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602)
 
 ## Troubleshooting
* Make sure to use the correct name of the environment
* Use the updated version of gym when running the pong code

   
