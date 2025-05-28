import rclpy
import numpy as np
import threading
import time
import os

from env import RobotEnv
from model_utils import Actor, Critic
from td3_agent import TD3
from replay_buffer import ReplayBuffer
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PointStamped
from utils import save_model, load_model

def sample_target_ur5e(base_position=np.array([0,0,0]), max_reach=0.8):
    while True:
        point = np.random.uniform(-max_reach, max_reach, 3) + base_position
        # Nokta robotun tabanından max_reach uzaklıkta olmalı
        if np.linalg.norm(point - base_position) <= max_reach and point[2] >= 0:
            return point


def main():
    rclpy.init()
    env = RobotEnv()
    executor = MultiThreadedExecutor()
    executor.add_node(env)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    target_pub = env.create_publisher(PointStamped, 'target_position', 10)

    # Model ve Replay Buffer oluşturuluyor
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

    # Checkpoint yükleme
    load_choice = input("Checkpoint'ten yüklemek istiyor musunuz? (e/h): ").strip().lower()
    if load_choice == 'e':
        checkpoints_dir = 'checkpoints'
        if not os.path.exists(checkpoints_dir):
            print("⚠️ 'checkpoints/' klasörü bulunamadı. Sıfırdan başlanacak.")
        else:
            checkpoints = sorted([f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')])
            if checkpoints:
                print("Mevcut Checkpointler:")
                for ckpt in checkpoints:
                    print(f"- {ckpt}")
                filename = input("Yüklemek istediğiniz checkpoint dosyasını girin (örnek: deneme1): ").strip()
                if not filename.endswith('.pth'):
                    filename += '.pth'
                full_path = os.path.join(checkpoints_dir, filename)
                if os.path.exists(full_path):
                    load_model(td3_agent, replay_buffer, full_path)  # .pth uzantısını kaldırma
                    print(f"✅ '{filename}' yüklendi.")
                else:
                    print(f"❌ Hata: '{full_path}' bulunamadı. Sıfırdan başlanacak.")
            else:
                print("⚠️ Kayıtlı checkpoint bulunamadı. Sıfırdan başlanacak.")

    # Eğitim döngüsü
    episodes = 1000
    best_reward = float('inf')
    ACTION_TIMEOUT = 20

    try:
        for ep in range(episodes):
            target_position = sample_target_ur5e()
            target_translation = [0, 0, 0, 0]

            point_msg = PointStamped()
            point_msg.header.frame_id = 'world'
            point_msg.header.stamp = env.get_clock().now().to_msg()
            point_msg.point.x, point_msg.point.y, point_msg.point.z = target_position

            print(f'\n--- Episode {ep+1} ---')
            print(f'Target Position: {target_position}')
            print(f'Target Translation: {target_translation}')

            state = env.reset(target_position, target_translation)
            total_reward = 0.0
            done = False
            step_count = 0

            while not done:
                iter_start = time.time()
                target_pub.publish(point_msg)
                step_count += 1
                print(f'İterasyon: {step_count}')

                action = td3_agent.select_action(np.array(state))
                print(f'Action: {action}')

                # Step denemesi
                while True:
                    try:
                        next_state, reward, done = env.step(action)
                        break
                    except TimeoutError:
                        if time.time() - iter_start > ACTION_TIMEOUT:
                            print(f"\u23f0 {ACTION_TIMEOUT} saniyeyi aştı. Episode iptal.")
                            env.teleport_to_home()
                            reward = 0.0
                            next_state = state
                            done = True
                            break
                        time.sleep(0.1)

                print(f'Next State: {next_state}')
                print(f'Reward: {reward}, Done: {done}')

                if done and reward == 0.0:
                    print("⚠️ Çarpışma! Episode erken sonlandırıldı.")
                    env.teleport_to_home()
                    break

                replay_buffer.add(state, action, reward, next_state, float(done))
                if replay_buffer.size() > 100:
                    td3_agent.train(replay_buffer)
                    print(f'Buffer size: {replay_buffer.size()}')

                state = next_state
                total_reward += reward

                if done:
                    print(f"🏁 Episode sona erdi. Robot eve dönüyor.")
                    env.teleport_to_home()
                    break

            print(f'Episode {ep+1} Total Reward: {total_reward:.2f}')

            # En iyi reward güncelle
            if total_reward < best_reward:
                best_reward = total_reward
                print(f'✅ Yeni en iyi reward: {best_reward:.2f}')
                os.makedirs("checkpoints", exist_ok=True)
                save_model(td3_agent, replay_buffer, f"checkpoints/best")

    except KeyboardInterrupt:
        print("\n🛑 Eğitim kullanıcı tarafından durduruldu.")
        save_choice = input("Checkpoint kaydedilsin mi? (e/h): ").strip().lower()
        if save_choice == 'e':
            os.makedirs("checkpoints", exist_ok=True)
            filename = input("Kaydetmek için dosya adı girin (uzantısız): ").strip()
            save_model(td3_agent, replay_buffer, f"checkpoints/{filename}")
            print(f"📦 Model kaydedildi: checkpoints/{filename}.pth")
        else:
            print("📭 Kaydetmeden çıkılıyor.")
    finally:
        env.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
