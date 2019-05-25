#agent_torch.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .utils import Transition, ReplayMemory
from .model_torch import DQN, DQN_Linear
import random

class DqnAgent():
    def __init__(self, 
                 state_shape, action_space,
                 device,
                 soft_update_ratio=0.01, 
                 learning_rate=1e-4, 
                 gamma=0.99, 
                 batch_size=128,
                 update_every = 10,
                 memory_size=10000,
                 conv_mode=True,
                 ):
        self.state_shape = state_shape,
        self.action_space = action_space
        self.soft_update_ratio = soft_update_ratio
        self.gamma = gamma
        self.update_every = update_every
        self.batch_size = batch_size
        self.is_conv_agent=conv_mode
        self.device = device
        
        # -- init -- #
        if conv_mode:
            self.q_policy_net = DQN(input_size=state_shape, output_size=action_space).to(device)
            self.q_target_net = DQN(input_size=state_shape, output_size=action_space).to(device)
        else:
            self.q_policy_net = DQN_Linear(input_size=state_shape, output_size=action_space).to(device)
            self.q_target_net = DQN_Linear(input_size=state_shape, output_size=action_space).to(device)

        self.q_target_net.load_state_dict(self.q_policy_net.state_dict()) # sync weights
        self.q_target_net.eval()
        
        self.optimizer = optim.RMSprop(self.q_policy_net.parameters(), lr=learning_rate)
        
        self.memory = ReplayMemory(capacity=memory_size)
        self.t_step = 0

        #self.loss_fn = nn.MSELoss()
        self.loss_fn = F.smooth_l1_loss
        
    def step(self, state, action, next_state, reward, is_done):
        """Add memory and learn"""
        state = state.astype('float32')
        next_state = next_state.astype('float32')
        self.memory.push(state, action, next_state, reward, is_done*1.)

        if len(self.memory) > self.batch_size:
            experience = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*experience))

            if self.is_conv_agent:
                state = torch.tensor(batch.state).permute(0, 3, 1, 2).to(self.device)
                next_state = torch.tensor(batch.next_state).permute(0, 3, 1, 2).to(self.device)
            else:
                state = torch.tensor(batch.state).to(self.device)
                next_state = torch.tensor(batch.next_state).to(self.device)
            action = torch.tensor(batch.action).to(self.device)
            reward = torch.tensor(batch.reward).to(self.device)
            is_done = torch.tensor(batch.done).to(self.device)

            q_targets_next = self.q_target_net(next_state).max(1)[0].detach() # Q(s', a)
            q_targets = reward + (self.gamma * q_targets_next * (1 - is_done)) # R + gamma*Q(s',a)

            q_expected = self.q_policy_net(state).gather(1, action.unsqueeze(1)) # Q(s,a)

            # optimize: l = R + gamma*Q(s', a) - Q(s, a)
            #loss = F.smooth_l1_loss(q_expected, q_targets.unsqueeze(1)) # batch x 1
            loss = self.loss_fn(q_expected, q_targets.unsqueeze(1))  # batch x 1

            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clip
            for params in self.q_policy_net.parameters():
                params.grad.data.clamp_(-3, 3)
            self.optimizer.step()

            self.t_step = (self.t_step + 1) % self.update_every
            if self.t_step == 0:
                self._soft_update(target_model=self.q_target_net,
                                  local_model=self.q_policy_net,
                                  tau=self.soft_update_ratio)
    
    def act(self, state, eps=0.):
        """Generate actions"""
        state = state.astype('float32')
        if random.random() > eps:
            if self.is_conv_agent:
                state = torch.tensor(state[np.newaxis, :,:,:]).permute(0, 3, 1, 2).to(self.device)
            else:
                state = torch.tensor(state[np.newaxis, :]).to(self.device)
            self.q_policy_net.eval() # swap to evaluation mode
            with torch.no_grad():
                action_values = self.q_policy_net(state)
            self.q_policy_net.train() # swap to training mode

            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_space))
    
    def _soft_update(self, target_model, local_model, tau):
        for target_params, local_params in zip(target_model.parameters(), local_model.parameters()):
            target_params.data.copy_(tau*local_params.data + (1.-tau)*target_params.data)