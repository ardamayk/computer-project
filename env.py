import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import TransformStamped
import tf2_ros
import numpy as np
from moveit_msgs.srv import GetStateValidity
import time

class RobotEnv(Node):
    def __init__(self):
        super().__init__('robot_env')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # JointTrajectory yayıncısı
        self.publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )
        # JointState aboneliği
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.current_joint_angles = None
        self.last_print_time = self.get_clock().now().seconds_nanoseconds()[0]

        # Eklem isimleri (UR5e için 6 eklem)
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        # Home (güvenli başlangıç) pozisyonu
        self.home_position = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]

        # Her episode için hedef pozisyon
        self.target_position = None
        self.target_translation = None

        # Çarpışma kontrolü için MoveIt servisi
        self.collision_client = self.create_client(
            GetStateValidity,
            '/check_state_validity'
        )
        while not self.collision_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Collision kontrol servisi bekleniyor...')

        # Ödül parametreleri
        self.reward_radius = 0.35   # 0.3 m yarıçaplı küre
        self.done_radius = 0.2    # 0.15 m içinde başarılı kabulü

        # Bir kez ödül bölgesine girip girmediğimizi izlemek için bayrak
        self.entered_reward_zone = False

    def get_end_effector_position(self):
        from_frame = "base_link"
        to_frame = "wrist_3_link"

        tf_timeout_sec = 5.0
        start = self.get_clock().now().seconds_nanoseconds()[0]
        while not self.tf_buffer.can_transform(from_frame, to_frame, rclpy.time.Time()):
            if self.get_clock().now().seconds_nanoseconds()[0] - start > tf_timeout_sec:
                self.get_logger().warn(f"TF alınamadı: {from_frame} → {to_frame} (timeout)")
                return None, None
            rclpy.spin_once(self, timeout_sec=0.1)

        try:
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                from_frame,
                to_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0)
            )
            translation = trans.transform.translation
            position = np.array([translation.x, translation.y, translation.z])
            rotation = trans.transform.rotation
            quaternion = np.array([rotation.x, rotation.y, rotation.z, rotation.w])
            return position, quaternion
        except Exception as e:
            self.get_logger().warn(f"TF dönüşümü alınamadı: {str(e)}")
            return None, None

    def joint_state_callback(self, msg):
        self.current_joint_angles = np.array(msg.position[:6])
        return self.current_joint_angles

    def compute_reward(self, old_position, current_position):

        position, _ = self.get_end_effector_position()
        raw_distance = np.linalg.norm(position - self.target_position)
        reward =  -5 * raw_distance  # Ceza katsayısı azaltıldı

        if old_position is None or current_position is None or self.target_position is None:
            return reward
        """
        - Eğer TF alınamazsa veya hedef yoksa: ağır ceza (-10.0).
        - raw_distance <= reward_radius: 
             entered_reward_zone bayrağını True yap, 
             pozitif ödül = (reward_radius - raw_distance).
        - raw_distance > reward_radius:
             eğer entered_reward_zone True ise: çift kat ceza = -2 * (raw_distance - reward_radius)
             değilse (henüz iç bölgeye girmemiş) normal ceza = - (raw_distance - reward_radius)
        """
        if position is None or self.target_position is None:
            return reward
        '''
        raw_distance = np.linalg.norm(position - self.target_position)

        if raw_distance <= self.reward_radius:
            # İç bölgeye ilk ya da tekrar giriş
            self.entered_reward_zone = True
            return 10 * (self.reward_radius - raw_distance)
        else:
            # Dış bölge
            diff = raw_distance - self.reward_radius
            if self.entered_reward_zone:
                return -20.0 * diff
            else:
                return -10.0 * diff
        '''


        old_distance = np.linalg.norm(old_position - self.target_position)
        new_distance = np.linalg.norm(current_position - self.target_position)

        if new_distance < old_distance:
            print("Hedefe yaklaşıyor")
        else:
            print("Hedeften uzaklaşıyor")

        reward = 10 * (old_distance - new_distance)  # Hedefe yaklaşınca pozitif


        # Hedefin yerden yüksekliği 0.2'nin altındaysa ve end-effector'un z'si 0.1 veya 0.05'in altına inerse ceza uygula

        return reward
    
    def get_observation(self, target_position, target_translation):
        position, rotation = self.get_end_effector_position()
        if self.current_joint_angles is None or position is None:
            return np.zeros(20)
        observation = np.concatenate((
            self.current_joint_angles,    # 6 boyut: eklem açıları
            position,                     # 3 boyut: uç efektör pozisyonu
            rotation,                     # 4 boyut: uç efektör rotasyonu (quaternion)
            target_position,              # 3 boyut: hedef pozisyon
            target_translation            # 4 boyut: hedef rotasyonu
        ))
        return observation  # Toplam 6+3+4+3+4 = 20 boyut

    def is_done(self):
        position, _ = self.get_end_effector_position()
        if position is None or self.target_position is None:
            return False
        raw_distance = np.linalg.norm(position - self.target_position)
        return raw_distance < self.done_radius

    def teleport_to_home(self):
        self.get_logger().info("Home pozisyonuna ışınlanıyor...")
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = self.home_position
        point.velocities = [0.0] * len(self.joint_names)
        point.accelerations = [0.0] * len(self.joint_names)
        point.time_from_start = Duration(sec=1, nanosec=0)
        msg.points.append(point)

        # Mevcut hareketi durdurmak için boş bir yörünge gönder
        stop_msg = JointTrajectory()
        stop_msg.joint_names = self.joint_names
        self.publisher.publish(stop_msg)
        rclpy.spin_once(self)

        # Home pozisyonuna komut gönder
        self.publisher.publish(msg)
        timeout_start = time.time()
        while time.time() - timeout_start < 2.0:
            rclpy.spin_once(self)
            if self.current_joint_angles is not None:
                error = np.linalg.norm(self.current_joint_angles - np.array(self.home_position))
                if error < 0.01:
                    self.get_logger().info("Home'a varıldı, 2 saniye bekleniyor...")
                    wait_start = time.time()
                    while time.time() - wait_start < 2.0:
                        rclpy.spin_once(self)
                    self.get_logger().info("Bekleme tamamlandı.")
                    return
        self.get_logger().warn("Home'a varma zaman aşımı!")
        if self.current_joint_angles is not None:
            self.current_joint_angles = np.array(self.home_position)

    def reset(self, target_position, target_translation):
        """
        Bölüm başında home pozisyonuna ışınlanır, entered_reward_zone bayrağı sıfırlanır,
        hedef belirlenir ve ilk gözlem döndürülür.
        """
        self.teleport_to_home()
        # Yeni bölüm başladığı için bayrağı False yap
        self.entered_reward_zone = False

        position, _ = self.get_end_effector_position()
        if position is not None:
            self.get_logger().info(f"End-effector: {position}")
        self.target_position = target_position
        self.target_translation = target_translation
        return self.get_observation(target_position, target_translation)

    def check_collision(self):
        if self.current_joint_angles is None:
            return False

        request = GetStateValidity.Request()
        request.robot_state.joint_state.name = self.joint_names
        request.robot_state.joint_state.position = self.current_joint_angles[:6].tolist()
        request.group_name = 'ur_manipulator'

        future = self.collision_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        if future.result() is not None:
            result = future.result()
            return not result.valid
        else:
            self.get_logger().error("Servis hatası, çarpışma varsayılıyor!")
            return True

    def step(self, action):
        """
        Aksiyon vektörünü alır, robotu o joint açılarına taşır:
        - Çarpışma durumunda: -10 ceza, home'a ışınla, bölüm bitir.
        - raw_distance <= done_radius ise: sadece done=True (ödül compute_reward'dan gelir).
        - raw_distance <= reward_radius: pozitif ödül (0.3 – raw_distance), bayrak True.
        - raw_distance > reward_radius:
            • entered_reward_zone True ise: ceza = -2*(raw_distance – 0.3)
            • entered_reward_zone False ise: ceza = -(raw_distance – 0.3)
        """
        if self.current_joint_angles is None:
            self.get_logger().warn("Joint açıları alınamadı.")
            return np.zeros(20 ), -10.0, True  # Bölümü hemen sonlandır

        old_position, _ = self.get_end_effector_position()
        new_joint_states = self.current_joint_angles + np.array(action)
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = new_joint_states.tolist()
        point.time_from_start = Duration(sec=1, nanosec=0)
        msg.points.append(point)
        self.publisher.publish(msg)
        self.get_logger().info(f"Aksiyon gönderildi: {point.positions}")

        last_collision_check = time.time()
        step_start_time = time.time()
        while True:
            if time.time() - step_start_time > 30.0:
                raise TimeoutError("Step zaman aşımı")

            # Periyodik çarpışma kontrolü
            if time.time() - last_collision_check >= 1.0:
                self.publisher.publish(msg)
                if self.check_collision():
                    # Çarpışma anında ağır ceza, home'a ışınla, bölümü sonlandır
                    reward = -10.0
                    self.teleport_to_home()
                    next_obs = self.get_observation(self.target_position, self.target_translation)
                    return next_obs, reward, True
                last_collision_check = time.time()

            rclpy.spin_once(self, timeout_sec=0.1)
            if self.current_joint_angles is None:
                continue
            error = np.linalg.norm(self.current_joint_angles - new_joint_states)
            if error < 0.01:
                break

        # Adım sonrası son çarpışma kontrolü
        if self.check_collision():
            reward = -10.0
            self.teleport_to_home()
            next_obs = self.get_observation(self.target_position, self.target_translation)
            return next_obs, reward, True

        # Ödül, gözlem, done
        new_position, _ = self.get_end_effector_position()
        obs = self.get_observation(self.target_position, self.target_translation)
        reward = self.compute_reward(old_position, new_position)
        done = self.is_done()

        if done:
            reward = 30  # Başarı ödülü artırıldı
            print("🎉 Hedefe ulaşıldı! 🎉")

        return obs, reward, done
