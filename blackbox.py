import rclpy
import numpy as np
import threading
from env import RobotEnv
from model_utils import Actor, Critic
from td3_agent import TD3
from model_io import load_checkpoint
from replay_buffer import ReplayBuffer
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PointStamped

def main():
    # ROS2 başlatma
    rclpy.init()
    env = RobotEnv()
    executor = MultiThreadedExecutor()
    executor.add_node(env)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    # Hedef pozisyon yayıncısı
    target_pub = env.create_publisher(PointStamped, 'target_position', 10)

    # Model parametreleri
    state_dim = 20
    action_dim = 6
    max_action = 0.2

    # Model ve ajan oluşturma
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

    # Replay buffer oluştur
    replay_buffer = ReplayBuffer()

    # Kullanıcıdan model ismi alma
    model_name = input("Kullanmak istediğiniz model adını girin: ").strip()
    done = False
    # Modeli yükle
    success, _ = load_checkpoint(td3_agent, replay_buffer, model_name)
    if not success:
        print("⚠️ Model yüklenemedi!")
        return

    try:
        while True:
            # Kullanıcıdan hedef nokta alma
            print("\nHedef noktayı girin (x y z) veya çıkmak için 'q':")
            user_input = input().strip()
            
            if user_input.lower() == 'q':
                break

            try:
                x, y, z = map(float, user_input.split())
                target_position = np.array([x, y, z])
            except ValueError:
                print("⚠️ Geçersiz giriş! Lütfen x y z formatında sayısal değerler girin.")
                continue

            # Hedef pozisyonu yayınla
            point_msg = PointStamped()
            point_msg.header.frame_id = 'world'
            point_msg.header.stamp = env.get_clock().now().to_msg()
            point_msg.point.x, point_msg.point.y, point_msg.point.z = target_position
            target_pub.publish(point_msg)

            # Ortamı sıfırla ve başlangıç durumunu al
            target_translation = [0, 0, 0, 0]  # Quaternion formatında rotasyon
            state = env.reset(target_position, target_translation)

            # Modelden aksiyon al
            while done is False:
                action = td3_agent.select_action(np.array(state))
                print(f"\nModel tarafından üretilen aksiyon: {action}")

                # Aksiyonu uygula
                try:
                    next_state, reward, done = env.step(action)
                    print(f"Reward: {reward:.2f}")
                    if done:
                        print("Hedefe ulaşıldı!")
                    else:
                        print("Hedefe ulaşılamadı.")
                except Exception as e:
                    print(f"⚠️ Hata oluştu: {str(e)}")
                
                state = next_state

            # Robotu başlangıç pozisyonuna döndür
            env.teleport_to_home()

    except KeyboardInterrupt:
        print("\nProgram kullanıcı tarafından durduruldu.")
    finally:
        # Executor'ı düzgün şekilde kapat
        executor.shutdown()
        executor_thread.join(timeout=1.0)
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 