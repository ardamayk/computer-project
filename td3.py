import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.layer(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa)

class ReplayBuffer:
    def __init__(self, max_size=int(1e6)):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.storage) < self.max_size:
            self.storage.append(data)
        else:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size=100):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch = [self.storage[i] for i in ind]

        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1)
        )

    def size(self):
        return len(self.storage)

class TD3:
    def __init__(self, state_dim, action_dim, max_action, actor, critic, actor_target, critic_target, policy_freq=2):
        self.actor = actor
        self.actor_target = actor_target
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = critic
        self.critic_target = critic_target
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.policy_freq = policy_freq
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).detach().cpu().numpy().flatten()

    def train(self, replay_buffer, batch_size=10, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5):
        self.total_it += 1

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (
                torch.randn_like(action) * policy_noise
            ).clamp(-noise_clip, noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            print(f'target1: {target_Q1} target2: {target_Q2}')
            target_Q = torch.max(target_Q1, target_Q2)
            target_Q = reward  + gamma * target_Q # bakilacak
            '''(1 - done) *'''
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
