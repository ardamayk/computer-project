import torch
import torch.nn as nn
import torch.optim as optim

class TD3:
    def __init__(self, state_dim, action_dim, max_action,
                 actor, critic, actor_target, critic_target,
                 policy_freq=2):
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
        """
        Çıkarım sırasında kullanılacak metot.
        Verilen state'i (numpy array) Torch tensörüne çevirir,
        actor’dan eylemi alır ve numpy array olarak döner.
        """
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).detach().cpu().numpy().flatten()

    def train(self, replay_buffer,
              batch_size=100, gamma=0.99, tau=0.005,
              policy_noise=0.2, noise_clip=0.5):
        """
        TD3’ün eğitim adımı:
         1) Replay buffer’dan rasgele batch al,
         2) Critic hedef değerlerini hesapla (target policy smoothing),
         3) Critic1 ve Critic2 ayrı ayrı güncelle (Double-Q Learning),
         4) Her policy_freq adımında actor’u ve hedef ağları (soft update) güncelle,
         5) Geri dönerken hem actor hem iki critic’in ayrı ayrı loss değerlerini return et.
        """

        self.total_it += 1

        # 1) Batch örneği al
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # 2) Critic hedef değerlerini hesapla (target policy smoothing)
        with torch.no_grad():
            # 2.1) Actor_target üzerinden bir sonraki eylemi al, eyleme gürültü ekle
            noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # 2.2) Critic_target’la Q1, Q2 değerlerini al
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            # 2.3) Minimumu seç, TD hedefini oluştur
            target_Q = torch.min(target_Q1, target_Q2)
            y = reward + (1 - done) * gamma * target_Q.squeeze()

        # 3) Critic güncellemesi
        current_Q1, current_Q2 = self.critic(state, action)
        current_Q1 = current_Q1.squeeze()
        current_Q2 = current_Q2.squeeze()

        # Ayrı ayrı MSE hesaplayalım:
        critic1_loss = nn.MSELoss()(current_Q1, y)
        critic2_loss = nn.MSELoss()(current_Q2, y)
        critic_loss = critic1_loss + critic2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_val = None
        # 4) Delayed Policy Güncellemesi (actor ve target ağları)
        if self.total_it % self.policy_freq == 0:
            # 4.1) Actor loss (Q1’i maximize etmek için negatif ortalama)
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 4.2) Soft update: Critic_target
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            # 4.3) Soft update: Actor_target
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            # Actor loss değerini sayıya çevir
            actor_loss_val = actor_loss.item()

        # Critic’lerin ayrı ayrı loss değerlerini sayıya çevir
        critic1_loss_val = critic1_loss.item()
        critic2_loss_val = critic2_loss.item()

        # 5) Geri dönerken actor & critic1 & critic2 loss değerlerini ver
        return actor_loss_val, critic1_loss_val, critic2_loss_val
