import rclpy # ROS2 Python istemci kütüphanesini içe aktar
import numpy as np # NumPy kütüphanesini içe aktar
import threading # Threading modülünü içe aktar
from env import RobotEnv # RobotEnv sınıfını env dosyasından içe aktar
from td3 import Actor, Critic, ReplayBuffer, TD3 # TD3 algoritması ile ilgili sınıfları td3 dosyasından içe aktar
from rclpy.executors import MultiThreadedExecutor # Çoklu iş parçacıklı yürütücüyü içe aktar

def main(): # Ana fonksiyonu tanımla
    rclpy.init() # ROS2'yi başlat
    env = RobotEnv() # RobotEnv sınıfından bir örnek oluştur

    # Start a multi-threaded executor in a separate thread # Çoklu iş parçacıklı bir yürütücüyü ayrı bir thread'de başlat
    executor = MultiThreadedExecutor() # MultiThreadedExecutor'dan bir örnek oluştur
    executor.add_node(env) # Ortam düğümünü yürütücüye ekle
    executor_thread = threading.Thread(target=executor.spin, daemon=True) # Yürütücü için bir thread oluştur (daemon olarak ayarla)
    executor_thread.start() # Thread'i başlat

    state_dim = 20 # Durum uzayının boyutu
    action_dim = 6 # Aksiyon uzayının boyutu
    max_action = 0.2 # Maksimum aksiyon değeri

    actor = Actor(state_dim, action_dim, max_action) # Aktör modelini oluştur
    actor_target = Actor(state_dim, action_dim, max_action) # Hedef aktör modelini oluştur
    actor_target.load_state_dict(actor.state_dict()) # Hedef aktörün ağırlıklarını aktörün ağırlıklarıyla başlat

    critic = Critic(state_dim, action_dim) # Critic modelini oluştur
    critic_target = Critic(state_dim, action_dim) # Hedef critic modelini oluştur
    critic_target.load_state_dict(critic.state_dict()) # Hedef critic'in ağırlıklarını critic'in ağırlıklarıyla başlat

    replay_buffer = ReplayBuffer() # Deneyim tekrarı buffer'ını oluştur

    td3_agent = TD3( # TD3 ajanını oluştur
        state_dim=state_dim, # Durum boyutu
        action_dim=action_dim, # Aksiyon boyutu
        max_action=max_action, # Maksimum aksiyon
        actor=actor, # Aktör modeli
        critic=critic, # Critic modeli
        actor_target=actor_target, # Hedef aktör modeli
        critic_target=critic_target, # Hedef critic modeli
        policy_freq=1 # Politika güncelleme frekansı
    )

    episodes = 1000 # Toplam bölüm sayısı
    best_reward = float('inf')  # Çünkü tüm reward'lar negatif # En iyi ödülü sonsuz olarak başlat (negatif ödüller için)

    for ep in range(episodes): # Her bölüm için döngü
        target_position = np.array([0.5, 0.2, 1]) # Hedef pozisyonu belirle
        target_translation = [0, 0, 0, 0] # Hedef translasyonu belirle (quaternion olarak düşünülmüş olabilir, ancak burada sadece 4 elemanlı bir liste)

        print(f'\n--- Episode {ep+1} ---') # Bölüm numarasını yazdır
        print(f'Target Position: {target_position}') # Hedef pozisyonu yazdır
        print(f'Target Translation: {target_translation}') # Hedef translasyonu yazdır

        state = env.reset(target_position, target_translation) # Ortamı sıfırla ve başlangıç durumunu al
        print(f'Initial State: {state}') # Başlangıç durumunu yazdır

        done = False # Bölümün bitip bitmediğini gösteren bayrak
        total_reward = 0 # Toplam ödülü sıfırla

        i = 0
        while not done: # Bölüm bitene kadar döngü
            i += 1
            print(f'iterasyon: {i}') # Durumu yazdır
            action = td3_agent.select_action(np.array(state)) # Ajanstan aksiyon seç
            print(f'Action: {action}') # Seçilen aksiyonu yazdır

            next_state, reward, done = env.step(action) # Ortamda bir adım at ve sonraki durumu, ödülü ve bitiş durumunu al
            print(f'Next State: {next_state}') # Sonraki durumu yazdır
            print(f'Reward: {reward}, Done: {done}') # Ödülü ve bitiş durumunu yazdır
            
            # Çarpışma durumunu kontrol et (reward 0.0 ve done True ise çarpışma olmuştur)
            if done and reward == 0.0:
                print("Çarpışma nedeniyle episode sonlandırıldı!")
                # Robotu direkt başlangıç konumuna ışınla
                env.teleport_to_home()
                break

            # Çarpışma olmadıysa deneyimi buffer'a ekle ve ajanı eğit
            replay_buffer.add(state, action, reward, next_state, float(done)) # Deneyimi buffer'a ekle

            if replay_buffer.size() > 25: # Eğer buffer'da yeterli deneyim varsa
                td3_agent.train(replay_buffer) # Ajanı eğit

            state = next_state # Durumu güncelle
            total_reward += reward # Toplam ödülü güncelle
            
            # Hedef tamamlandıysa da robotu başlangıç konumuna ışınla
            if done:
                print(f"Hedef tamamlandı! Episode sona erdi.")
                env.teleport_to_home()
                break

        print(f'Episode {ep+1} Total Reward: {total_reward:.2f}') # Bölümün toplam ödülünü yazdır

        if total_reward < best_reward: # Eğer mevcut bölümün ödülü en iyi ödülden daha iyiyse (daha az negatifse)
            best_reward = total_reward # En iyi ödülü güncelle
            print(f'✅ New Best Reward: {best_reward:.2f}') # Yeni en iyi ödülü yazdır

    # Shutdown # Kapatma işlemleri
    env.destroy_node() # Ortam düğümünü yok et
    rclpy.shutdown() # ROS2'yi kapat
    # executor_thread.join() kaldırıldı # Yürütücü thread'inin birleştirilmesi kaldırıldı

if __name__ == '__main__': # Eğer dosya doğrudan çalıştırılıyorsa
    main() # Ana fonksiyonu çağır
