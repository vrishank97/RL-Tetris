import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

num_episodes = 60000
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 1000
TARGET_UPDATE = 10

import matplotlib
import matplotlib.pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()

if is_ipython:
    from IPython import display


episode_durations = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, h, w):
        super(DQN, self).__init__()
        #self.pool = nn.AvgPool2d(3, stride=4)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        print(linear_input_size)
        self.fullconnected = nn.Linear(linear_input_size,32)
        self.head = nn.Linear(32, 12)

    def forward(self, x):
        #x = F.relu(self.bn1(self.conv1(self.pool(x))))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fullconnected(x))
        return self.head(x)


policy_net = DQN(22, 42).to(device)

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(75000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(12)]], device=device, dtype=torch.long)

resize = T.Compose([T.ToPILImage(),
                    T.Resize(22, interpolation=Image.CUBIC),
                    T.ToTensor()])  

def preproc(screen):
    screen = screen.transpose((2, 1, 0))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    screen = resize(screen).unsqueeze(0).to(device)
    return screen

def optimize_model():
    if len(memory) < 1000:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    #print(batch.next_state[0])

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = policy_net(non_final_next_states).max(1)[0].detach()
    #print(next_state_values)
    reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


import gym_tetris
env = gym_tetris.make('Tetris-v0')
done = True
#env.render()

for i in range(num_episodes):
    state = env.reset()
    #print(state.shape)
    state = preproc(state[0:420, 0:220])
    #print(state.shape)
    #state = torch.tensor([state], device=device, dtype=torch.float32)
    for step in range(5000):
        action = select_action(state)
        next_state, reward, done, info = env.step(action)
        if done:
            reward=-1
            print("Dead at {}".format(step))
        else:
            reward= 1
        '''
        if step%500==0:
            img = Image.fromarray(next_state, 'RGB')
            img.show()
        '''
        #print(next_state.shape)
        next_state = preproc(next_state[0:420, 0:220])
        #print(next_state.shape)
        #print(next_state)
        #next_state = torch.tensor([next_state], device=device, dtype=torch.float32)
        reward = torch.tensor([reward], device=device)
        if done:
            next_state = None
        memory.push(state, action, next_state, reward)
        state = next_state

        if done:
            break
    print("Episode: {}/{}, score: {}".format(i, num_episodes, info))

    optimize_model()
