
import cv2
import pyscreeze
import numpy as np
import time
import pyautogui
import os
import random
from mss import mss
import random
#from keras import Sequentialn
from collections import deque
#from keras.layers import Dense
#import matplotlib.pyplot as plt
#from keras.optimizers import Adam
#from plot_script import plot_result
from PIL import Image
#from tkinter import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

import torch.optim as optim

from collections import namedtuple, deque

w=40
h=40
ws=15
gen=1

#координаты где нажали мышкой
# root=Tk()
# def callback(event):
#     print(event.x, event.y)
# root.bind('<Button-1>', callback)
# root.mainloop()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Snake():
    def __init__(self):
        self.color=[50,50,50] #тело змеи (хз)
        self.color2=[255,0,0] #яблоко (красный)
        self.color3=[85,255,0] #голова змеи (зеленый)
        self.left=0
        self.top=0
        self.state_snake=0
        self.old_reward=0
        self.i=0

    #создание массива с полем змейкой и яблоком
    def coord_snake(self):
        #скрин экрана
        pyautogui.screenshot('screenshot.png', region=(620,310,600,600))
        im=Image.open('screenshot.png')
        self.snake = np.zeros((40, 40))
        self.snake_x=np.zeros(1600)
        self.snake_y=np.zeros(1600)
        self.apple_x=0
        self.apple_y=0
        self.left=0
        self.top=0
        states=0
        self.state=np.zeros(4, dtype=float)
        for x in range(40):
                
                if x>0:
                    self.left=0
                    self.top+=ws
                for y in range(40):
                    
                    a=self.left+ws/2
                    b=self.top+ws/2
                    obj1 = im.getpixel((a, b))
                    #t=im.crop((self.left,self.top, self.left+ws,self.top+ws))
                    #t.show()
                    #print (obj1)
                    obj=[]
                    for i in obj1:
                        obj.append(i)

                    if np.array_equal(obj,self.color):
                        a+=1
                        self.snake[x,y]=1
                        self.snake_x[a]=x
                        self.snake_y[a]=y
                    if np.array_equal(obj,self.color2):
                        self.snake[x,y]=2
                        self.apple_x=x
                        self.apple_y=y
                    if np.array_equal(obj,self.color3):
                        self.snake[x,y]=3
                        self.snake_x[0]=x
                        self.snake_y[0]=y
                        self.state_snake=states
                    self.left+=ws
                    states+=1       

        self.state[0]=self.snake_x[0]
        self.state[1]=self.snake_y[0]

        self.state[2]=self.apple_x
        self.state[3]=self.apple_y
        # self.coord['snake']=self.snake

       # self.state=[self.snake_x[0],self.snake_y[0],self.apple_x[0],self.apple_y[0]]
        return self.state

    def get_state(self):
        pass

    def step(self, action):
        """Принимает действие, 
        возвращает состояние после действия, награду
        и информацию об окончании эпизода"""
        if action==0:
            pyautogui.press('left')
        if action==1:
            pyautogui.press('right')
        if action==2:
            pyautogui.press('up')
        if action==3:
            pyautogui.press('down')
        old_state=self.state
        state=self.coord_snake()
        r=self.reward()
        if (state[0]==19 and state[1]==20 and (old_state[0]!=18 and old_state[0]!=20 and old_state[1]!=19 and old_state[1]!=21)):
            done=False
            #r-=500
        else:
            done=True
        

        return state, r, done

    def reward(self):
        """Возвращает награду"""
        self.distance=np.sqrt(np.power(self.snake_x[0]-self.apple_x,2)+
        np.power(self.snake_y[0]-self.apple_y,2))
        reward=(1/self.distance)*20
        if (self.old_reward>reward):
            reward/=10
        # else:
        #     reward*=2
        # if self.state[0]==self.state[2] and self.state[1]==self.state[2]:
        #     reward+=600
        if self.i==20:
            self.i=0
            reward-=200


        self.old_reward=reward
        return reward

    def reset(self):
        """Сбрасывает среду к стартовому состоянию 
        и возвращает первое действие(рандомно)"""
        pyautogui.press('shift')
        #new_coord=self.coord_snake()
        #done=True
        #action=random.randint(0,2)
        state=self.coord_snake()
        return state

    
    def action(self):
        pass


class DQN(nn.Module):

    def __init__(self, state_size, action_size, seed):
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, h):
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, state_size, action_size, seed=1):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = DQN(state_size, action_size, seed).to(device)
        self.qnetwork_target = DQN(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        """
        """
        states, actions, rewards, next_states, dones = experiences

        # get max predicted q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def act(self, state, eps=0.1):
        state = torch.from_numpy(state).float().to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    
class ReplayBuffer:
    """Буфер фиксированного размера(хранит Q таблицу)"""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Params
        ======
            action_size (int): размерность каждого действия
            buffer_size (int): максимальный размер буфера
            batch_size (int): размер каждой обучающей партии
            seen (int): случайное семя
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


#Функция жадности
def createEpsilonGreedyPolicy(Q, epsilon, num_actions):
    """
    Создает эпсилон-жадную стратегию
    на заданную Q-функцию и эпсилон.

    Возвращает функцию, которая принимает состояние
    в качестве входных данных и возвращает вероятности
    за каждое действие в виде массива
    длины пространства действия (множество возможных действий).
    """
    def policyFunction(state):
        #вероятность события
        # if np.random.random() < epsilon:
            #return np.random.choice(num_actions)
    # Else choose the action with the highest value
        # else:
        #     return np.argmax(Q[state])
            Action_probabilities = np.ones(num_actions,
            dtype = float) * epsilon / num_actions
            best_action = np.argmax(Q[state])
            Action_probabilities[best_action] += (1.0 - epsilon)
            return Action_probabilities
            

    return policyFunction  

#Функция запуска новых поколений
def train_snake(gen, env):
    size_states=4
    
    size_actions=4
    agent=Agent(size_states, size_actions, 15)

    #словарь, который отображает состояние-(действие-значение действия)

    Q = defaultdict(lambda: np.zeros(size_actions))

    num_episodes=500
    #функция жадности
    #policy = createEpsilonGreedyPolicy(Q, epsilon=0.2, size_actions=4)
    #для каждого эпизода
    scores = []
    apples = []
    reward=0
    score=0
    for i in range(num_episodes):
        #сбросить окружение и выбрать первое действие
        done = True
        print("Score: " + str(score))
        print("Reward: "+ str(reward))
        reward=0
        score=0
        state = env.reset()
        print (i)

        while done:
            # получить вероятности всех действий из текущего состояния
            action = agent.act(state)

            # выбрать действие согласно распределению вероятностей
            # action = np.random.choice(np.arange(
            #           len(action)),
            #            p = action)
            # принять меры и получить награду, перейти в следующее состояние
            next_state, reward, done = env.step(action)

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            scores.append(score)               # save most recent score
            
            # best_next_action = np.argmax(Q[next_state])    
            # discount_factor=0.9
            # td_target = reward + discount_factor * Q[next_state][best_next_action]

            # td_delta = td_target - Q[state][action]
            # alpha = 1
            # Q[state][action] += alpha * td_delta
            # #time.sleep(2)
            # # done - False, если эпизод завершен
            # if done==False:
            #     print (reward)
            #     break
            
    return Q



if __name__ == '__main__':
    params = dict()
 

    env=Snake()

    
    Q=train_snake(500, env)

   
    
