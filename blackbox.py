import rclpy
import numpy as np
import threading
from env import RobotEnv
from model_utils import Actor, Critic
from td3_agent import TD3
from model_io import load_checkpoint, save_checkpoint, list_available_models
from replay_buffer import ReplayBuffer
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PointStamped
import os
import time
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Duration

class CollisionException(Exception):
    pass

def main():
    # ROS2 başlatılır
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

    # Model ve ajan oluşturulur
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

    # Replay buffer oluşturulur
    replay_buffer = ReplayBuffer()

    # Kullanıcıya checkpointten yükleme isteyip istemediği sorulur
    load_choice = input("Checkpoint'ten yüklemek istiyor musunuz? (e/h): ").strip().lower()
    if load_choice == 'e':
        model_dirs = list_available_models()
        if model_dirs:
            print("Mevcut Modeller:")
            for model_dir in model_dirs:
                print(f"- {model_dir}")
            model_name = input("Yüklemek istediğiniz model adını girin: ").strip()
            success, _ = load_checkpoint(td3_agent, replay_buffer, model_name)
            if not success:
                print("⚠️ Model yüklenemedi. Sıfırdan başlanacak.")
        else:
            print("⚠️ Kayıtlı model bulunamadı. Sıfırdan başlanacak.")
    else:
        model_name = input("Kullanmak istediğiniz model adını girin: ").strip()
        success, _ = load_checkpoint(td3_agent, replay_buffer, model_name)
        if not success:
            print("⚠️ Model yüklenemedi!")
            return

    try:
        while True:
            # Kullanıcıdan hedef nokta alınır
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
            done = False

            # Robot hedefe ulaşana veya çarpışana kadar döngü
            while not done:
                action = td3_agent.select_action(np.array(state))
                print(f"\nModel tarafından üretilen aksiyon: {action}")

                try:
                    next_state, done = step(env, action)
                    if done:
                        print("Hedefe ulaşıldı! Robot duruyor.") 
                    state = next_state
                except Exception as e:
                    print(f"⚠️ Hata oluştu: {str(e)}")
                    env.teleport_to_home()
                    break

            # Hedefe ulaşsa da ulaşmasa da yeni hedef bekleniyor

    except KeyboardInterrupt:
        print("\nProgram kullanıcı tarafından durduruldu.")
    finally:
        executor.shutdown()
        executor_thread.join(timeout=1.0)
        env.destroy_node()
        rclpy.shutdown()

def step(env, action):
    """
    Aksiyon vektörünü alır, robotu o joint açılarına taşır.
    Çarpışma veya zaman aşımı durumunda özel exception fırlatır.
    """
    print("Anlık eklem açıları:", env.current_joint_angles)

    new_joint_states = env.current_joint_angles + np.array(action)

    msg = JointTrajectory()
    msg.joint_names = env.joint_names
    point = JointTrajectoryPoint()
    point.positions = new_joint_states.tolist()
    point.time_from_start = Duration(sec=1, nanosec=0)
    msg.points.append(point)
    env.publisher.publish(msg)

    last_collision_check = time.time()
    step_start_time = time.time()
    done = False
    while True:
        # Zaman aşımı kontrolü
        if time.time() - step_start_time > 3.0:
            raise TimeoutError("Step zaman aşımı")

        # Periyodik çarpışma kontrolü
        if time.time() - last_collision_check >= 1.0:
            env.publisher.publish(msg)
            if env.check_collision():
                # Çarpışma tespit edildiğinde exception fırlat
                raise CollisionException("Çarpışma tespit edildi!")
            last_collision_check = time.time()

        rclpy.spin_once(env, timeout_sec=0.1)
        if env.current_joint_angles is None:
            continue
        
        error = np.linalg.norm(env.current_joint_angles - new_joint_states)
        if error < 0.01:
            obs = env.get_observation(env.target_position, env.target_translation)
            done = env.is_done()
            return obs, done

        done = env.is_done()
        if done:
            obs = env.get_observation(env.target_position, env.target_translation)
            return obs, done

 

if __name__ == '__main__':
    main() 