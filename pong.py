import numpy as np
import gym
import cv2
import random


env = gym.make("ALE/Pong-v5", render_mode="human")


episodes = 500
alpha=0.001
gamma = 0.99
epsilon = 1.0
decay = 0.995
epsilonmin = 0.1
batchsize = 32
max_size = 10000


replay=[]
weights = np.random.rand(85 * 85, 50)
bias = np.random.rand(50)

def convolve(resized,filter):
 nheight=resized.shape()
 nwidth=resized.shape()
 fheight=filter.shape()
 fwidth=filter.shape()
 stride=3
 new=np.pad(resized,width=1)
 p=2
 outputwidth=(nwidth-2*p+fwidth)//stride
 outputheight=(nheight-2*p+fheight)//stride
 output=np.zeros(outputwidth,outputheight)
 for i in range(outputwidth):
    for j in range(outputheight):
        output = np.sum(input[i:i + fheight, j:j + fwidth] * filter)


def pooling(input_image):
    pool = 2
    stride = 2

    nheight, nwidth = input_image.shape

    outputheight = (nheight - pool) // stride + 1
    outputwidth = (nwidth - pool) // stride + 1

    output = np.zeros((outputheight, outputwidth))

    for i in range(outputheight):
        for j in range(outputwidth):
            output[i, j] = np.max(input_image[i * stride:i * stride + pool, j * stride:j * stride + pool])

    return output
def activation(x):
    return np.maximum(0, x)
def fullyconnected(input_vector, weights, bias):
    return np.dot(input_vector, weights) + bias
def softmax(x):
    exp=np.exp(x - np.max(x))
    return exp/ np.sum(exp)
def cnnforward(input_image):
    f1 = np.random.rand(5, 5)
    c = convolve(input_image, f1)
    a = activation(c)
    p = pooling(a)
    flat = p.flatten()
    fc = fullyconnected(flat, weights, bias)
    res = softmax(fc)
    return res

def loss(prediction, target):
    return np.sum((target - prediction))


def cnn_backprop(input_image, target,alpha=0.01):
    global weights, bias

    f1 = np.random.rand(5, 5)
    c = convolve(input_image, f1)
    a = activation(c)
    p = pooling(a)
    flat = p.flatten()
    fc=fullyconnected(flat, weights, bias)
    prediction = softmax(fc)
    value = loss(prediction, target)
    gradientsoftmax = prediction - target
    gradientw, gradientb, gradientflat = fcbackprop(flat, weights, bias, gradientsoftmax)
    weights, bias = update(weights, bias, gradientw, gradientb,alpha)


    pooled = gradientflat.reshape(p.shape)


    gradientactivation = poolingbackprop(p, pooled)
    gradientconvolve = activation_backprop(a, gradientactivation)

    return value, weights, bias


def update(w, bias, gradientweights, gradientbias, alpha):
    w -= alpha * gradientweights
    bias -= alpha * gradientbias
    return w, bias


def fcbackprop(flat, w, bias, gradientoutput):
    gradientweights = np.dot(flat.T, gradientoutput)
    gradientbias = np.sum(gradientoutput, axis=0)
    gradientinput = np.dot(gradientoutput, w.T)
    return gradientweights, gradientbias, gradientinput


def poolingbackprop(pooled, output):
    pool = 2
    stride = 2
    nheight, nwidth = pooled.shape
    input = np.zeros_like(pooled)

    for i in range(nheight):
        for j in range(nwidth):
            region = pooled[i * stride:i * stride + pool, j * stride:j * stride + pool]
            max=np.argmax(region)
            input[i * stride + max[0], j * stride + max[1]]=output[i, j]

    return input




def activation_backprop(output, gradient):
    grad_input = np.zeros_like(gradient)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if output[i, j] > 0:
                input[i, j]=output[i, j]
    return grad_input


def selection(q_values, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_values)

def batchtraining(batch, gamma):
    global weights, bias

    for state, action, reward, next_state, done in batch:

        gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        resizedstate = cv2.resize(gray, (85, 85))


        graynext = cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY)
        resizedstatenew = cv2.resize(graynext, (85, 85))


        qvalues = cnnforward(resizedstate)
        nextqvalues = cnnforward(resizedstatenew)


        target = reward
        if not done:
            target += gamma * np.max(nextqvalues)

        loss =loss(qvalues[action], target)
        grad_softmax = qvalues - target
        grad_softmax[:]=0
        grad_softmax[action] = qvalues[action] - target

        grad_weights, grad_bias, _ = fcbackprop(resizedstate.flatten(), weights, bias, grad_softmax)
        weights, bias = update(weights, bias, grad_weights, grad_bias,alpha)
def train():

     for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:

            gray_state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            resized_state = cv2.resize(gray_state, (85, 85))
            q_values = cnnforward(resized_state)


            action=selection(q_values, epsilon)
            next_state, reward, done, _ = env.step(action)
            replay.append((state, action, reward, next_state, done))


            if len(replay) >=batchsize:
                minibatch = random.sample(replay, batchsize)
                batchtraining(minibatch, gamma)

            state = next_state
            total_reward += reward

        epsilon = max(epsilonmin, epsilon * decay)



train()


