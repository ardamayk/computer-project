import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import TransformStamped
import tf2_ros 
import numpy as np

class RobotEnv(Node):
    def __init__(self):
        super().__init__('robot_env')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.publisher = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.subscription = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.current_joint_angles = None

        self.joint_names = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]

        self.link_names = [
            "shoulder_link", "upper_arm_link", "forearm_link",
            "wrist_1_link", "wrist_2_link", "wrist_3_link"
        ]

        self.target_position = None
        self.target_translation = None


    def get_end_effector_position(self):
        from_frame = "base_link"
        to_frame = "wrist_3_link"

        # TF mesajı gelene kadar bekle (maksimum 5 saniye)
        tf_timeout_sec = 5.0
        start = self.get_clock().now().seconds_nanoseconds()[0]
        while not self.tf_buffer.can_transform(from_frame, to_frame, rclpy.time.Time()):
            if self.get_clock().now().seconds_nanoseconds()[0] - start > tf_timeout_sec:
                self.get_logger().warn(f"TF mesajı alınamadı: {from_frame} → {to_frame} (timeout)")
                return None, None
            rclpy.spin_once(self, timeout_sec=0.1)

        try:
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                from_frame, to_frame, rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0)
            )

            translation = trans.transform.translation
            position = np.array([translation.x, translation.y, translation.z])

            rotation = trans.transform.rotation
            quaternion = np.array([rotation.x, rotation.y, rotation.z, rotation.w])

            #self.get_logger().info(f"{from_frame} → {to_frame}: {position}")
            return position, quaternion

        except Exception as e:
            self.get_logger().warn(f"TF dönüşümü alınamadı: {from_frame} → {to_frame}. Hata: {str(e)}")
            return None, None

            
           
    def joint_state_callback(self, msg):
        tf_timeout_sec = 5.0  # Timeout süresi
        start_time = self.get_clock().now().seconds_nanoseconds()[0]

        # Gelen mesajı kontrol et ve state'i güncelle
        self.current_joint_angles = np.array(msg.position[:6])  # İlk 6 eklem

        # Eğer joint açıları henüz alınmadıysa, bekle
        while self.current_joint_angles is None:
            if self.get_clock().now().seconds_nanoseconds()[0] - start_time > tf_timeout_sec:
                self.get_logger().warn("Joint state mesajı alınamadı (timeout).")
                return None  # Timeout durumu, None döner
            rclpy.spin_once(self, timeout_sec=0.1)  # 0.1 saniye bekleyip tekrar kontrol et

        # Gelen mesajı kullandıktan sonra logla
        return self.current_joint_angles

    def quaternion_distance(self, q1, q2):
        dot = np.abs(np.dot(q1, q2))
        return 2 * np.arccos(np.clip(dot, -1.0, 1.0))


        
    def compute_reward(self):
        position, rotation = self.get_end_effector_position()
        if position is None or self.target_position is None:
            return -100.0  # Ceza
        # Pozisyon farkı
        pos_diff = 10 * (position - self.target_position)  # (3,)

        # Rotasyon farkı, örneğin quaternion farkı için ayrı işlenmeli
        rot_diff = self.quaternion_distance(rotation, self.target_translation)  # Skaler fark

        # Toplam ödül metriği
        distance = np.linalg.norm(pos_diff) #+ rot_diff)

        reward = -distance  # Hedefe daha yakın olmanın ödülünü artırıyoruz.
        return reward

    
    def get_observation(self, target_position, target_translation):
        position, rotation = self.get_end_effector_position()

        if self.current_joint_angles is None or position is None:
            return np.zeros(20)  # 20 boyutlu sıfır dizisi döndür

        # 20 boyutlu observation
        observation = np.concatenate((self.current_joint_angles, position, rotation, target_position, target_translation))
        return observation


    def is_done(self):
        position, rotation = self.get_end_effector_position()
        if position is None or self.target_position is None:
            return True
        pos_diff = 10 * (position - self.target_position)  # (3,)

        # Rotasyon farkı, örneğin quaternion farkı için ayrı işlenmeli
        rot_diff = self.quaternion_distance(rotation, self.target_translation) # Skaler fark

        # Toplam ödül metriği
        distance = np.linalg.norm(pos_diff)#+ rot_diff)
        return np.linalg.norm(distance) < 0.5

    def reset(self, target_position, target_translation):
        
        self.target_position = target_position
        self.target_translation = target_translation


        current_state = self.get_observation(target_position, target_translation)
        return current_state

    def step(self, action):
        if self.current_joint_angles is None:
            self.get_logger().warn("Joint açıları henüz alınmadı.")
            return np.zeros(20), -100.0, True

        new_joint_states = np.array(self.current_joint_angles) + np.array(action)

        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = new_joint_states.tolist()
        point.time_from_start = Duration(sec=2, nanosec=0)

        msg.points.append(point)
        self.publisher.publish(msg)
        self.get_logger().info(f'Yeni aksiyon gönderildi: {point.positions}')

        while True:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.current_joint_angles is None:
                continue
            error = np.linalg.norm(np.array(self.current_joint_angles) - new_joint_states)
            if error < 0.01:
                break

        obs = self.get_observation(self.target_position, self.target_translation)
        reward = self.compute_reward()
        done = self.is_done()

        return obs, reward, done
