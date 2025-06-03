import matplotlib.pyplot as plt
import os

class RewardVisualizer:
    def __init__(self):
        plt.ion()  # İnteraktif mod
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title('Eğitim Ödülleri')
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Toplam Ödül')
        self.ax.legend()
        self.ax.grid(True)
        
    def update(self, current_episodes, current_rewards, previous_episodes=None, previous_rewards=None):
        try:
            self.ax.clear()
            self.ax.set_title('Eğitim Ödülleri')
            self.ax.set_xlabel('Episode')
            self.ax.set_ylabel('Toplam Ödül')
            
            # Önceki eğitim reward'larını çiz (mavi)
            if previous_episodes is not None and previous_rewards is not None and len(previous_episodes) > 0:
                self.ax.plot(previous_episodes, previous_rewards, 'b--', label='Önceki Eğitim')
            
            # Mevcut eğitim reward'larını çiz (yeşil)
            if current_episodes is not None and current_rewards is not None and len(current_episodes) > 0:
                self.ax.plot(current_episodes, current_rewards, 'g-', label='Mevcut Eğitim')
            
            self.ax.legend()
            self.ax.grid(True)
            plt.draw()
            plt.pause(0.1)
        except Exception as e:
            print(f"Plot güncelleme hatası: {e}")
            
    def close(self):
        plt.close('all') 