import matplotlib.pyplot as plt
import os

class RewardVisualizer:
    def __init__(self):
        plt.ion()  # İnteraktif moda geç
        # İki alt eksen: Üst → Reward, Alt → Loss (actor + critic1 + critic2)
        self.fig, (self.ax_r, self.ax_l) = plt.subplots(
            nrows=2, ncols=1, figsize=(10, 8), sharex=True
        )

        # Üst grafik: Reward
        self.ax_r.set_title('Eğitim Ödülleri')
        self.ax_r.set_ylabel('Toplam Ödül')
        self.ax_r.grid(True)

        # Alt grafik: Loss (actor, critic1, critic2)
        self.ax_l.set_title('Ağ Kayıpları')
        self.ax_l.set_xlabel('Episode')
        self.ax_l.set_ylabel('Loss')
        self.ax_l.grid(True)

    def update(self,
               current_episodes, current_rewards,
               current_actor_losses, current_critic1_losses, current_critic2_losses,
               previous_episodes=None, previous_rewards=None,
               previous_actor_losses=None, previous_critic1_losses=None, previous_critic2_losses=None):
        """
        Hem ödülleri hem de actor/critic1/critic2 loss’larını günceller.
        previous_... parametreleri checkpoint’ten geliyorsa önceki grafikler de çizilir.
        """

        try:
            # ===== ÜST: Reward Grafiği =====
            self.ax_r.clear()
            self.ax_r.set_title('Eğitim Ödülleri')
            self.ax_r.set_ylabel('Toplam Ödül')
            self.ax_r.grid(True)

            if previous_episodes is not None and previous_rewards is not None and len(previous_episodes) > 0:
                self.ax_r.plot(previous_episodes, previous_rewards, 'b--', label='Önceki Eğitim')
            if current_episodes is not None and current_rewards is not None and len(current_episodes) > 0:
                self.ax_r.plot(current_episodes, current_rewards, 'g-', label='Mevcut Eğitim')
            self.ax_r.legend()

            # ===== ALT: Loss Grafiği =====
            self.ax_l.clear()
            self.ax_l.set_title('Ağ Kayıpları')
            self.ax_l.set_xlabel('Episode')
            self.ax_l.set_ylabel('Loss')
            self.ax_l.grid(True)

            # Önceki Eğitim’den: Actor, Critic1, Critic2 loss’ları
            if (previous_episodes is not None and previous_actor_losses is not None and
                previous_critic1_losses is not None and previous_critic2_losses is not None and
                len(previous_episodes) == len(previous_actor_losses) and
                len(previous_episodes) == len(previous_critic1_losses) and
                len(previous_episodes) == len(previous_critic2_losses) and
                len(previous_episodes) > 0):
                # Actor (mavi nokta), Critic1 (turuncu nokta), Critic2 (mor nokta)
                self.ax_l.plot(previous_episodes, previous_actor_losses,
                               'b--', label='Önceki Actor Loss')
                self.ax_l.plot(previous_episodes, previous_critic1_losses,
                               'c--', label='Önceki Critic1 Loss')
                self.ax_l.plot(previous_episodes, previous_critic2_losses,
                               'm--', label='Önceki Critic2 Loss')

            # Mevcut Eğitim’den: Actor, Critic1, Critic2 loss’ları
            if (current_episodes is not None and current_actor_losses is not None and
                current_critic1_losses is not None and current_critic2_losses is not None and
                len(current_episodes) == len(current_actor_losses) and
                len(current_episodes) == len(current_critic1_losses) and
                len(current_episodes) == len(current_critic2_losses) and
                len(current_episodes) > 0):
                self.ax_l.plot(current_episodes, current_actor_losses,
                               'g-', label='Mevcut Actor Loss')
                self.ax_l.plot(current_episodes, current_critic1_losses,
                               'r-', label='Mevcut Critic1 Loss')
                self.ax_l.plot(current_episodes, current_critic2_losses,
                               'k-', label='Mevcut Critic2 Loss')

            self.ax_l.legend()

            # Güncellemeyi çiz ve kısa bekleme
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)

        except Exception as e:
            print(f"Plot güncelleme hatası: {e}")

    def close(self):
        plt.close('all')
