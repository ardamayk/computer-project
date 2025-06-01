import matplotlib.pyplot as plt
import json
import os
import threading
import queue
import time

class RewardVisualizer:
    def __init__(self):
        self.current_rewards = []
        self.previous_rewards = []
        self.command_queue = queue.Queue()
        self.running = True
        
        # Plot pencerelerini başlat
        plt.ion()
        self.fig1 = plt.figure(1, figsize=(10, 6))  # Mevcut eğitim için
        self.fig2 = plt.figure(2, figsize=(10, 6))  # Tüm eğitimler için
        
        # Thread'i başlat
        self.thread = threading.Thread(target=self._visualization_loop, daemon=True)
        self.thread.start()

    def _visualization_loop(self):
        """Görselleştirme thread'i"""
        while self.running:
            try:
                # Kuyruktan komut al
                command = self.command_queue.get(timeout=0.1)
                if command['type'] == 'update':
                    self._update_plots(command['current_rewards'], command['previous_rewards'])
                elif command['type'] == 'close':
                    self.running = False
                    break
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Görselleştirme hatası: {e}")

    def _update_plots(self, current_rewards, previous_rewards):
        """Plot'ları güncelle"""
        # Mevcut eğitim plot'u
        plt.figure(1)
        plt.clf()
        plt.plot(current_rewards, label='Mevcut Eğitim', color='blue')
        plt.title('Mevcut Eğitim - Episode Reward Değerleri')
        plt.xlabel('Episode')
        plt.ylabel('Toplam Reward')
        plt.legend()
        plt.grid(True)
        
        # Tüm eğitimler plot'u
        plt.figure(2)
        plt.clf()
        
        # Önceki eğitimlerin rewardlarını çiz
        if previous_rewards:
            for i, rewards in enumerate(previous_rewards):
                plt.plot(rewards, label=f'Önceki Eğitim {i+1}', color='gray', alpha=0.3)
        
        # Mevcut eğitimin rewardlarını çiz
        plt.plot(current_rewards, label='Mevcut Eğitim', color='blue')
        
        plt.title('Tüm Eğitimler - Episode Reward Değerleri')
        plt.xlabel('Episode')
        plt.ylabel('Toplam Reward')
        plt.legend()
        plt.grid(True)
        
        # Her iki plot'u da güncelle
        plt.figure(1).canvas.draw()
        plt.figure(2).canvas.draw()
        plt.pause(0.1)

    def update(self, current_rewards, previous_rewards):
        """Plot'ları güncellemek için komut gönder"""
        self.command_queue.put({
            'type': 'update',
            'current_rewards': current_rewards,
            'previous_rewards': previous_rewards
        })

    def close(self):
        """Görselleştirmeyi kapat"""
        self.command_queue.put({'type': 'close'})
        self.thread.join()
        plt.close('all')

def load_previous_rewards():
    """Önceki eğitimlerden kaydedilmiş reward değerlerini yükler"""
    if os.path.exists('rewards_history.json'):
        with open('rewards_history.json', 'r') as f:
            return json.load(f)
    return []

def save_rewards(rewards):
    """Reward değerlerini JSON dosyasına kaydeder"""
    with open('rewards_history.json', 'w') as f:
        json.dump(rewards, f)

def update_plot(current_rewards, previous_rewards):
    plt.ion()
    plt.figure(1)
    plt.clf()
    plt.plot(current_rewards, label='Mevcut Eğitim', color='blue')
    plt.title('Mevcut Eğitim - Episode Reward Değerleri')
    plt.xlabel('Episode')
    plt.ylabel('Toplam Reward')
    plt.legend()
    plt.grid(True)
    plt.pause(0.1)

    plt.figure(2)
    plt.clf()
    # Önceki eğitimlerin rewardlarını çiz
    if previous_rewards:
        for i, rewards in enumerate(previous_rewards):
            plt.plot(rewards, label=f'Önceki Eğitim {i+1}', color='gray', alpha=0.3)
    plt.plot(current_rewards, label='Mevcut Eğitim', color='blue')
    plt.title('Tüm Eğitimler - Episode Reward Değerleri')
    plt.xlabel('Episode')
    plt.ylabel('Toplam Reward')
    plt.legend()
    plt.grid(True)
    plt.pause(0.1)

def init_plots():
    """Plot pencerelerini başlatır"""
    plt.ion()
    plt.figure(1, figsize=(10, 6))  # Mevcut eğitim için
    plt.figure(2, figsize=(10, 6))  # Tüm eğitimler için

def close_plots():
    """Plot pencerelerini kapatır"""
    plt.close('all') 