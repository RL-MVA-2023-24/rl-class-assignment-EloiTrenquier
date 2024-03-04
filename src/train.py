from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import trange
import gymnasium as gym
import pickle as pkl
import plotly.graph_objects as go
from copy import deepcopy
import random


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
) 

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        # ind_start = np.random.randint(0,len(self.data) - batch_size - 1)
        # batch = self.data[ind_start:ind_start+batch_size]
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
    


class ProjectAgent():
    def __init__(self, config={}, model=None):
        device = "cpu"
        state_dim, nb_neurons, n_action = 6, 256, 4
        pi = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                                nn.ReLU(),
                                nn.Linear(nb_neurons, nb_neurons),
                                nn.ReLU(), 
                                nn.Linear(nb_neurons, nb_neurons),
                                nn.ReLU(), 
                                nn.Linear(nb_neurons, nb_neurons),
                                nn.ReLU(), 
                                nn.Linear(nb_neurons, nb_neurons),
                                nn.ReLU(),  
                                nn.Linear(nb_neurons, n_action)).to(device)
        self.nb_actions = config['nb_actions'] if 'nb_actions' in config.keys() else 4
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.98
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 800
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.001
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 20000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 100
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop #0.0000125#
        self.model = model if model is not None else pi
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.SmoothL1Loss()#else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.01
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 400
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        
    def greedy_action(self, network, state):
        device = "cuda" if next(network.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()
        
    def test(self, env):
        s,_ = env.reset()
        results = []
        rewards = []
        actions = []
        for i in range(199):
            a=self.act(s)
            actions.append(a)
            s, r, done, trunc, _ = env.step(a)
            results.append(s)
            rewards.append(r)
        print(np.sum(rewards))
        return results, rewards, actions
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        bestsofar=0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict + (1-tau)*target_state_dict
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                if episode > 1 and episode%1==0:
                    validation_score = evaluate_HIV(agent=self, nb_episode=1)
                else :
                    validation_score = 0
                if validation_score > bestsofar:
                    bestsofar = validation_score
                    self.save("bestmodel")
                print(f"Ep. {episode}, eps {np.round(epsilon,4)}, batch size {min(len(self.memory), self.batch_size)}, return: {np.round(episode_cum_reward/1e6,2)}e6")
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
        self.save("ilbifteak_preconv")
        return episode_return
    
    def act(self, state, use_random=False):
        if use_random:
            action = env.action_space.sample()
        else: 
            #Greedy action
            action = self.greedy_action(self.model, state)
        return action

    def save(self, path="ilbifteak"):
        torch.save(self.model.state_dict(), path+".pt")

    def load(self):
        path = "ilbifteak.pt"
        device = torch.device('cpu')
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.eval()
    
    def load_from_path(self, path="ilbifteak.pt"):
        device = torch.device('cpu')
        self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.eval()
        
    def load_memory_from_path(self, path="sweet_memory.pkl"):
        with open(path, mode='rb') as file:
            self.memory = pkl.load(file)
            
    def dump_memory(self, path="sweet_memory.pkl"):
        with open(path, mode='wb') as file:
            pkl.dump(self.memory, file)