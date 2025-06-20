import rclpy
import numpy as np
import threading
import time
import os
from datetime import datetime

from env import RobotEnv
from model_utils import Actor, Critic
from td3_agent import TD3
from replay_buffer import ReplayBuffer
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from visualization import RewardVisualizer
from model_io import load_checkpoint, save_checkpoint, list_available_models

def sample_target_ur5e(base_position=np.array([0,0,0]), max_reach=0.8, z_min=None, z_max=None):
    """
    Kullanıcının tarif ettiği matematiksel yönteme göre nokta üretir.
    """

    while True:
        z = np.random.uniform(z_min, z_max)
        x_kalan_mesafe = np.sqrt(max_reach**2 - z**2)
        x = np.random.uniform(-x_kalan_mesafe, x_kalan_mesafe)
        y_kalan_mesafe = np.sqrt(x_kalan_mesafe**2 - x**2)
        y = np.random.uniform(-y_kalan_mesafe, y_kalan_mesafe)
        point = base_position + np.array([x, y, z])
        return point

def sample_random_quaternion():
    """
    Rastgele bir birim quaternion üretir.
    """
    axis = np.random.normal(0, 1, 3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(0, 2.0*np.pi)
    qw = np.cos(angle/2.0)
    qx = axis[0] * np.sin(angle/2.0)
    qy = axis[1] * np.sin(angle/2.0)
    qz = axis[2] * np.sin(angle/2.0)
    return np.array([qx, qy, qz, qw])

def create_orientation_marker(position, quaternion):
    """
    Tek bir marker (id=0) döndürür, böylece RViz'de her seferinde aynı ID'ye
    sahip ok güncellenir ve eski okumalar silinmez, yalnızca güncellenir.
    """
    marker = Marker()
    marker.header.frame_id = 'world'
    marker.header.stamp = rclpy.time.Time().to_msg()
    marker.ns = 'target_orientation'
    marker.id = 0           # Sabit ID: 0. Böylece tek bir ok güncellenir.
    marker.type = Marker.ARROW
    marker.action = Marker.ADD

    # Pozisyon ve oryantasyon
    marker.pose.position.x = float(position[0])
    marker.pose.position.y = float(position[1])
    marker.pose.position.z = float(position[2])
    marker.pose.orientation.x = float(quaternion[0])
    marker.pose.orientation.y = float(quaternion[1])
    marker.pose.orientation.z = float(quaternion[2])
    marker.pose.orientation.w = float(quaternion[3])

    # Ölçeklendirme: ok kalınlığı ve uzunluğu
    marker.scale.x = 0.2   # ok uzunluğu
    marker.scale.y = 0.05  # ok çapı
    marker.scale.z = 0.05  # ok çapı

    # Renk: yeşil ok
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    return marker

def main():
    rclpy.init()
    env = RobotEnv()
    executor = MultiThreadedExecutor()
    executor.add_node(env)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # RViz'e hem hedef pozisyonu hem de orientation marker yayınlanacak
    target_pub = env.create_publisher(PointStamped, 'target_position', 10)
    marker_pub = env.create_publisher(Marker, 'visualization_marker', 10)

    # Model ve replay buffer oluşturma
    state_dim = 20
    action_dim = 6
    max_action = 0.2

    actor = Actor(state_dim, action_dim, max_action)
    actor_target = Actor(state_dim, action_dim, max_action)
    actor_target.load_state_dict(actor.state_dict())

    critic = Critic(state_dim, action_dim)
    critic_target = Critic(state_dim, action_dim)
    critic_target.load_state_dict(critic.state_dict())

    td3_agent = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        actor=actor,
        critic=critic,
        actor_target=actor_target,
        critic_target=critic_target,
        policy_freq=1
    )

    replay_buffer = ReplayBuffer()

    # Episode başına biriktirilecek listeler
    current_rewards = []
    current_actor_losses = []
    current_critic1_losses = []
    current_critic2_losses = []
    current_episodes = []

    previous_episodes = None
    previous_rewards = None
    previous_actor_losses = None
    previous_critic1_losses = None
    previous_critic2_losses = None

    # Checkpoint yükleme
    load_choice = input("Checkpoint'ten yüklemek istiyor musunuz? (e/h): ").strip().lower()
    if load_choice == 'e':
        model_dirs = list_available_models()
        if model_dirs:
            print("Mevcut Modeller:")
            for model_dir in model_dirs:
                print(f"- {model_dir}")
            model_name = input("Yüklemek istediğiniz model adını girin: ").strip()
            success, (previous_episodes, previous_rewards) = load_checkpoint(td3_agent, replay_buffer, model_name)
            if not success:
                print("⚠️ Model yüklenemedi. Sıfırdan başlanacak.")
            else:
                # Önceki loss dosyalarını da CSV'den yüklemeye çalış
                try:
                    import pandas as pd
                    model_dir = os.path.join("checkpoints", model_name)
                    actor_losses_path = os.path.join(model_dir, "actor_losses.csv")
                    critic1_losses_path = os.path.join(model_dir, "critic1_losses.csv")
                    critic2_losses_path = os.path.join(model_dir, "critic2_losses.csv")

                    if os.path.exists(actor_losses_path):
                        df_a = pd.read_csv(actor_losses_path)
                        previous_actor_losses = df_a['actor_loss'].values.tolist()
                    if os.path.exists(critic1_losses_path):
                        df_c1 = pd.read_csv(critic1_losses_path)
                        previous_critic1_losses = df_c1['critic1_loss'].values.tolist()
                    if os.path.exists(critic2_losses_path):
                        df_c2 = pd.read_csv(critic2_losses_path)
                        previous_critic2_losses = df_c2['critic2_loss'].values.tolist()
                except Exception as e:
                    print(f"⚠️ Önceki loss dosyaları yüklenirken hata: {e}")
                    previous_actor_losses = None
                    previous_critic1_losses = None
                    previous_critic2_losses = None
        else:
            print("⚠️ Kayıtlı model bulunamadı. Sıfırdan başlanacak.")

    # Görselleştiriciyi başlat
    visualizer = RewardVisualizer()

    episodes = 10000
    best_reward = float('inf')
    ACTION_TIMEOUT = 20

    # Keşif gürültüsü ayarları
    initial_sigma = 0.1 * max_action
    final_sigma   = 0.01 * max_action
    decay_steps   = 50000
    total_steps = 0

    try:
        for ep in range(episodes):
            # 1) Rastgele pozisyon ve oryantasyon örnekle
            if ep < 750:
                # z >= 0.5
                target_position = sample_target_ur5e(max_reach=0.8, z_min=0.4, z_max=0.8)
            elif ep < 1500:
                # 0 <= z < 0.5
                target_position = sample_target_ur5e(max_reach=0.8, z_min=0.0, z_max=0.4)
            elif ep < 2500:
                # z < 0
                target_position = sample_target_ur5e(max_reach=0.8, z_min=-0.8, z_max=0.0)
            else:
                # tüm uzay
                target_position = sample_target_ur5e(max_reach=0.8, z_min=-0.8, z_max=0.8)

            print(f"Verilen hedef pozisyonu: {target_position}")
            target_quaternion = sample_random_quaternion()  # [qx, qy, qz, qw]

            # 2) RViz için PointStamped
            point_msg = PointStamped()
            point_msg.header.frame_id = 'world'
            point_msg.header.stamp = env.get_clock().now().to_msg()
            point_msg.point.x, point_msg.point.y, point_msg.point.z = target_position

            # 3) RViz için orientation Marker (id=0 sabit)
            marker = create_orientation_marker(target_position, target_quaternion)
            marker.header.stamp = env.get_clock().now().to_msg()

            print(f'\n--- Episode {ep+1} ---')

            # 4) Ortama reset: Pozisyon + Oryantasyon
            state = env.reset(target_position, target_quaternion.tolist())
            total_reward = 0.0
            done = False
            step_count = 0
            episode_start_time = time.time()  # Episode sayacını başlat

            # Episode boyunca biriktirilecek loss listeleri
            episode_actor_losses = []
            episode_critic1_losses = []
            episode_critic2_losses = []

            # 5) RViz'e hedef pozisyon ve markeri yayınla
            target_pub.publish(point_msg)
            marker_pub.publish(marker)

            while not done:
                iter_start = time.time()
                step_count += 1
                print(f'İterasyon: {step_count}')

                # 6.1) TD3'den ham eylem al
                raw_action = td3_agent.select_action(np.array(state))

                # 6.2) Exploration noise ekle
                frac = min(1.0, total_steps / decay_steps)
                sigma = initial_sigma * (1 - frac) + final_sigma * frac
                noise = np.random.normal(0, sigma, size=action_dim)
                action = (raw_action + noise).clip(-max_action, max_action)

                # 6.3) Ortamda bir adım ilerle, Timeout kontrolü
                while True:
                    try:
                        next_state, reward, done = env.step(action)
                        break
                    except TimeoutError:
                        if time.time() - iter_start > ACTION_TIMEOUT:
                            print(f"\u23f0 {ACTION_TIMEOUT} saniyeyi aştı. Episode iptal.")
                            env.teleport_to_home()
                            reward = -10.0
                            next_state = state
                            done = True
                            break
                        time.sleep(0.1)

                print(f'Reward: {reward:.2f}, Done: {done}')

                if done and reward == 0.0:
                    print("️ Çarpışma! Episode erken sonlandırıldı.")
                    env.teleport_to_home()
                    break

                # 6.4) ReplayBuffer'a ekle
                replay_buffer.add(state, action, reward, next_state, float(done))

                # 6.5) Öğrenme ve loss hesaplama
                if replay_buffer.size() > 100:
                    actor_loss_val, critic1_loss_val, critic2_loss_val = td3_agent.train(replay_buffer)
                    print(f'Buffer size: {replay_buffer.size()}')

                    if actor_loss_val is not None:
                        episode_actor_losses.append(actor_loss_val)
                    episode_critic1_losses.append(critic1_loss_val)
                    episode_critic2_losses.append(critic2_loss_val)

                    print(f'  > Actor Loss: {actor_loss_val if actor_loss_val is not None else 0.0:.4f}, '
                          f'Critic1 Loss: {critic1_loss_val:.4f}, Critic2 Loss: {critic2_loss_val:.4f}')

                state = next_state
                total_reward += reward
                total_steps += 1

                if done:
                    print(f"🏁 Episode sona erdi. Robot eve dönüyor.")
                    env.teleport_to_home()
                    break

            # Episode bitti, süreye bağlı ödülü hesapla ve ekle
            duration = time.time() - episode_start_time
            if ep >= 2500:
                time_reward = 15 - duration  # 15 saniyeden hızlıysa ödül, yavaşsa ceza
                print(f'Episode süresi: {duration:.2f}s, Zaman ödülü: {time_reward:.2f}')
                total_reward += time_reward
            else:
                print(f'Episode süresi: {duration:.2f}s, Zaman ödülü: 0.00 (ep < 2500)')

            print(f'Episode {ep+1} Total Reward: {total_reward:.2f}')

            # 7) Episode Sonu: Ortalama loss hesaplama
            if len(episode_actor_losses) > 0:
                avg_actor_loss = float(np.mean(episode_actor_losses))
            else:
                avg_actor_loss = 0.0

            if len(episode_critic1_losses) > 0:
                avg_critic1_loss = float(np.mean(episode_critic1_losses))
            else:
                avg_critic1_loss = 0.0

            if len(episode_critic2_losses) > 0:
                avg_critic2_loss = float(np.mean(episode_critic2_losses))
            else:
                avg_critic2_loss = 0.0

            # Episode numarası (önceki yüklemeden devam)
            if previous_episodes is not None and len(previous_episodes) > 0:
                current_episode_number = previous_episodes[-1] + len(current_episodes) + 1
            else:
                current_episode_number = len(current_episodes)
            current_episodes.append(current_episode_number)

            current_rewards.append(total_reward)
            current_actor_losses.append(avg_actor_loss)
            current_critic1_losses.append(avg_critic1_loss)
            current_critic2_losses.append(avg_critic2_loss)

            # 8) Her 500 epizotta otomatik checkpoint oluştur (farklı bir dizin adıyla)
            if (current_episode_number + 1) % 500 == 0:
                # Benzersiz bir isim için zaman damgası ekleyelim
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                auto_name = f"auto_ep{current_episode_number+1}_{timestamp}"
                print(f"🔖 Otomatik checkpoint kaydediliyor: {auto_name}")
                
                # Önceki ve mevcut değerleri birleştir
                combined_episodes = []
                combined_actor_losses = []
                combined_critic1_losses = []
                combined_critic2_losses = []
                combined_rewards = []

                # Önceki değerleri ekle (eğer varsa)
                if previous_episodes is not None:
                    combined_episodes.extend(previous_episodes)
                    if previous_actor_losses is not None:
                        combined_actor_losses.extend(previous_actor_losses)
                    if previous_critic1_losses is not None:
                        combined_critic1_losses.extend(previous_critic1_losses)
                    if previous_critic2_losses is not None:
                        combined_critic2_losses.extend(previous_critic2_losses)
                    if previous_rewards is not None:
                        combined_rewards.extend(previous_rewards)

                # Mevcut değerleri ekle
                combined_episodes.extend(current_episodes)
                combined_actor_losses.extend(current_actor_losses)
                combined_critic1_losses.extend(current_critic1_losses)
                combined_critic2_losses.extend(current_critic2_losses)
                combined_rewards.extend(current_rewards)
                
                save_checkpoint(td3_agent, replay_buffer, combined_rewards, auto_name,
                              combined_episodes, combined_actor_losses,
                              combined_critic1_losses, combined_critic2_losses)

            # 10) Grafikleri Güncelle
            visualizer.update(
                current_episodes, current_rewards,
                current_actor_losses, current_critic1_losses, current_critic2_losses,
                previous_episodes, previous_rewards,
                previous_actor_losses, previous_critic1_losses, previous_critic2_losses
            )
            print(f" Görselleştirme güncellendi. Toplam {len(current_rewards)} episode.")

    except KeyboardInterrupt:
        print("\n Eğitim kullanıcı tarafından durduruldu.")
        save_choice = input("Checkpoint ve değerler kaydedilsin mi? (e/h): ").strip().lower()
        if save_choice == 'e':
            model_name = input("Kaydetmek için model adı girin: ").strip()
            
            # Önceki ve mevcut değerleri birleştir
            combined_episodes = []
            combined_actor_losses = []
            combined_critic1_losses = []
            combined_critic2_losses = []
            combined_rewards = []

            # Önceki değerleri ekle (eğer varsa)
            if previous_episodes is not None:
                combined_episodes.extend(previous_episodes)
                if previous_actor_losses is not None:
                    combined_actor_losses.extend(previous_actor_losses)
                if previous_critic1_losses is not None:
                    combined_critic1_losses.extend(previous_critic1_losses)
                if previous_critic2_losses is not None:
                    combined_critic2_losses.extend(previous_critic2_losses)
                if previous_rewards is not None:
                    combined_rewards.extend(previous_rewards)

            # Mevcut değerleri ekle
            combined_episodes.extend(current_episodes)
            combined_actor_losses.extend(current_actor_losses)
            combined_critic1_losses.extend(current_critic1_losses)
            combined_critic2_losses.extend(current_critic2_losses)
            combined_rewards.extend(current_rewards)
            
            save_checkpoint(td3_agent, replay_buffer, combined_rewards, model_name,
                          combined_episodes, combined_actor_losses,
                          combined_critic1_losses, combined_critic2_losses)
        else:
            print(" Kaydetmeden çıkılıyor.")
    finally:
        visualizer.close()
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
