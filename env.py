import rclpy # ROS2 Python istemci kütüphanesini içe aktar
from rclpy.node import Node # ROS2 Düğüm (Node) sınıfını içe aktar
from builtin_interfaces.msg import Duration # Zaman süresi mesaj tipini içe aktar
from sensor_msgs.msg import JointState # Eklem durumları mesaj tipini içe aktar
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # Eklem yörüngesi ve yörünge noktası mesaj tiplerini içe aktar
from geometry_msgs.msg import TransformStamped # Dönüşüm (transformasyon) damgalı mesaj tipini içe aktar
import tf2_ros # ROS2 için TF2 (transformasyon kütüphanesi) modülünü içe aktar
import numpy as np # NumPy kütüphanesini np takma adıyla içe aktar

class RobotEnv(Node): # Robot ortamını temsil eden sınıf, rclpy.node.Node sınıfından miras alır
    def __init__(self): # Sınıfın yapıcı (constructor) metodu
        super().__init__('robot_env') # Üst sınıfın (Node) yapıcı metodunu 'robot_env' düğüm adıyla çağırır
        self.tf_buffer = tf2_ros.Buffer() # TF2 dönüşümlerini saklamak için bir arabellek (buffer) oluşturur
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self) # TF2 dönüşümlerini dinlemek için bir dinleyici (listener) oluşturur
        
        self.publisher = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10) # Eklem yörüngesi mesajlarını yayınlamak için bir yayıncı (publisher) oluşturur
        self.subscription = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10) # Eklem durumlarını dinlemek için bir abone (subscriber) oluşturur ve callback fonksiyonunu atar
        self.current_joint_angles = None # Mevcut eklem açılarını saklamak için bir değişken (başlangıçta None)

        self.joint_names = [ # Kontrol edilecek eklemlerin adlarını içeren bir liste
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", # Omuz ve dirsek eklemleri
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint" # Bilek eklemleri
        ]

        self.link_names = [ # Robotun link (bağlantı) adlarını içeren bir liste (şu anki kodda doğrudan kullanılmıyor gibi görünüyor)
            "shoulder_link", "upper_arm_link", "forearm_link", # Omuz, üst kol ve ön kol linkleri
            "wrist_1_link", "wrist_2_link", "wrist_3_link" # Bilek linkleri
        ]

        self.target_position = None # Hedef pozisyonu saklamak için bir değişken
        self.target_translation = None # Hedef rotasyonu/translasyonu saklamak için bir değişken (quaternion bekleniyor olabilir)


    def get_end_effector_position(self): # Robotun uç elemanının (end-effector) pozisyonunu ve rotasyonunu alır
        from_frame = "base_link" # Başlangıç referans çerçevesi (genellikle robotun tabanı)
        to_frame = "wrist_3_link" # Hedef referans çerçevesi (uç elemanın bağlı olduğu link)

        # TF mesajı gelene kadar bekle (maksimum 5 saniye)
        tf_timeout_sec = 5.0 # TF mesajı için zaman aşımı süresi (saniye cinsinden)
        start = self.get_clock().now().seconds_nanoseconds()[0] # Beklemeye başlama zamanını saniye olarak alır
        while not self.tf_buffer.can_transform(from_frame, to_frame, rclpy.time.Time()): # Belirtilen çerçeveler arasında dönüşüm yapılıp yapılamadığını kontrol eder
            if self.get_clock().now().seconds_nanoseconds()[0] - start > tf_timeout_sec: # Eğer zaman aşımına uğradıysa
                self.get_logger().warn(f"TF mesajı alınamadı: {from_frame} → {to_frame} (timeout)") # Uyarı mesajı loglar
                return None, None # Pozisyon ve rotasyon için None döndürür
            rclpy.spin_once(self, timeout_sec=0.1) # ROS2 callback'lerini işlemek için kısa bir süre bekler

        try: # Dönüşüm alma işlemini dener
            trans: TransformStamped = self.tf_buffer.lookup_transform( # Belirtilen çerçeveler arasındaki en son dönüşümü alır
                from_frame, to_frame, rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0) # Dönüşüm için zaman ve süre parametreleri
            )

            translation = trans.transform.translation # Dönüşümden translasyon (pozisyon) bilgisini alır
            position = np.array([translation.x, translation.y, translation.z]) # Translasyonu NumPy dizisine çevirir (x, y, z)

            rotation = trans.transform.rotation # Dönüşümden rotasyon (quaternion) bilgisini alır
            quaternion = np.array([rotation.x, rotation.y, rotation.z, rotation.w]) # Rotasyonu NumPy dizisine çevirir (x, y, z, w)

            #self.get_logger().info(f"{from_frame} → {to_frame}: {position}") # Pozisyon bilgisini loglamak için yorum satırı (isteğe bağlı)
            return position, quaternion # Hesaplanan pozisyon ve quaternion'u döndürür

        except Exception as e: # Dönüşüm alınırken bir hata oluşursa
            self.get_logger().warn(f"TF dönüşümü alınamadı: {from_frame} → {to_frame}. Hata: {str(e)}") # Hata mesajını loglar
            return None, None # Pozisyon ve rotasyon için None döndürür

            
           
    def joint_state_callback(self, msg): # Eklem durumları mesajı geldiğinde çağrılan callback fonksiyonu
        tf_timeout_sec = 5.0  # Timeout süresi # (Bu değişken burada kullanılmıyor gibi görünüyor, belki de current_joint_angles kontrolü için düşünülmüştü)
        start_time = self.get_clock().now().seconds_nanoseconds()[0] # (Bu değişken de aşağıdaki while döngüsü için mantıklı)

        # Gelen mesajı kontrol et ve state'i güncelle
        self.current_joint_angles = np.array(msg.position[:6])  # Gelen mesajdaki ilk 6 eklemin pozisyonunu NumPy dizisi olarak alır ve saklar

        # Eğer joint açıları henüz alınmadıysa, bekle (Bu döngü current_joint_angles'ın None olup olmadığını kontrol etmeli, ancak yukarıda hemen atanıyor)
        # Bu döngü muhtemelen current_joint_angles'ın None olma durumunu değil, belirli bir değere ulaşmasını beklemek için tasarlanmış olabilir ama mevcut haliyle direkt atanıyor.
        # Mantıksal olarak, eğer current_joint_angles callback çağrıldığında güncelleniyorsa, bu while döngüsü gereksizdir.
        # Eğer amaç ilk joint state mesajını beklemekse, __init__ içinde None olarak başlatılıp burada güncellendikten sonra bir flag ile kontrol edilebilir.
        while self.current_joint_angles is None: # Bu koşul yukarıdaki atama nedeniyle genellikle hemen false olur
            if self.get_clock().now().seconds_nanoseconds()[0] - start_time > tf_timeout_sec: # Zaman aşımı kontrolü
                self.get_logger().warn("Joint state mesajı alınamadı (timeout).") # Uyarı mesajı loglar
                return None  # Timeout durumu, None döner
            rclpy.spin_once(self, timeout_sec=0.1)  # 0.1 saniye bekleyip tekrar kontrol et

        # Gelen mesajı kullandıktan sonra logla (Loglama yapılmıyor, sadece değer döndürülüyor)
        return self.current_joint_angles # Güncellenmiş mevcut eklem açılarını döndürür

    def quaternion_distance(self, q1, q2): # İki quaternion arasındaki mesafeyi (açısal farkı) hesaplar
        dot = np.abs(np.dot(q1, q2)) # İki quaternion'un iç çarpımının mutlak değerini alır
        # dot değeri sayısal hatalar nedeniyle 1.0'dan biraz büyük veya -1.0'dan biraz küçük olabilir, bu nedenle klip edilir.
        return 2 * np.arccos(np.clip(dot, -1.0, 1.0)) # Açısal mesafeyi radyan cinsinden hesaplar


        
    def compute_reward(self): # Ajanın aldığı ödülü hesaplar
        position, rotation = self.get_end_effector_position() # Uç elemanın mevcut pozisyonunu ve rotasyonunu alır
        if position is None or self.target_position is None: # Eğer mevcut pozisyon veya hedef pozisyon alınamadıysa
            return -100.0  # Büyük bir ceza değeri döndürür
        # Pozisyon farkı
        pos_diff = 10 * (position - self.target_position)  # Mevcut pozisyon ile hedef pozisyon arasındaki farkı hesaplar ve 10 ile ölçeklendirir (3 boyutlu vektör)

        # Rotasyon farkı, örneğin quaternion farkı için ayrı işlenmeli
        # rot_diff = self.quaternion_distance(rotation, self.target_translation)  # Mevcut rotasyon ile hedef rotasyon arasındaki açısal farkı hesaplar (skaler)
        # Yukarıdaki satır yorumlu olduğu için rotasyon farkı ödüle dahil edilmiyor.

        # Toplam ödül metriği
        distance = np.linalg.norm(pos_diff) # Sadece pozisyon farkının L2 normunu (Öklid mesafesini) hesaplar
                                        # Yorum satırındaki `+ rot_diff` eklenirse rotasyon farkı da dahil edilir.

        reward = -distance  # Ödülü, hedefe olan mesafenin negatifi olarak tanımlar (mesafe azaldıkça ödül artar/sıfıra yaklaşır)
        return reward # Hesaplanan ödülü döndürür

    
    def get_observation(self, target_position, target_translation): # Ajan için gözlem (observation) uzayını oluşturur
        position, rotation = self.get_end_effector_position() # Uç elemanın mevcut pozisyonunu ve rotasyonunu alır

        if self.current_joint_angles is None or position is None: # Eğer mevcut eklem açıları veya pozisyon bilgisi yoksa
            # Bu durum genellikle başlangıçta veya bir hata durumunda oluşabilir.
            return np.zeros(20)  # 20 boyutlu bir sıfır dizisi döndürür (gözlem boyutuna uygun olmalı)

        # 20 boyutlu observation
        # Gözlem; mevcut eklem açıları, uç eleman pozisyonu, uç eleman rotasyonu, hedef pozisyon ve hedef translasyonu/rotasyonu içerir.
        # Boyutlar: eklem açıları (6), pozisyon (3), rotasyon (4), hedef_pozisyon (3), hedef_translasyon (4) = 6+3+4+3+4 = 20
        observation = np.concatenate((self.current_joint_angles, position, rotation, target_position, target_translation))
        return observation # Oluşturulan gözlem dizisini döndürür


    def is_done(self): # Bölümün (episode) bitip bitmediğini kontrol eder
        position, rotation = self.get_end_effector_position() # Uç elemanın mevcut pozisyonunu ve rotasyonunu alır
        if position is None or self.target_position is None: # Eğer mevcut pozisyon veya hedef pozisyon alınamadıysa
            return True # Bölümü sonlandır (hata durumu veya başlangıç durumu olabilir)
        pos_diff = 10 * (position - self.target_position)  # Mevcut pozisyon ile hedef pozisyon arasındaki farkı hesaplar ve ölçeklendirir

        # Rotasyon farkı, örneğin quaternion farkı için ayrı işlenmeli
        # rot_diff = self.quaternion_distance(rotation, self.target_translation) # Rotasyon farkını hesaplar (şu an kullanılmıyor)

        # Toplam ödül metriği (Burada 'distance' aslında pozisyon farkının normu)
        distance = np.linalg.norm(pos_diff) # Sadece pozisyon farkının L2 normunu hesaplar
                                            # Yorum satırındaki `+ rot_diff` eklenirse rotasyon farkı da dahil edilir.
        # Hedefe yeterince yaklaşıldıysa bölümü sonlandırır.
        return np.linalg.norm(distance) < 0.5 # Pozisyonel mesafenin normu 0.5'ten küçükse True döndürür (hedefe ulaşıldı kabul edilir)
                                             # np.linalg.norm(distance) ifadesi, distance zaten bir skaler olduğu için gereksiz, sadece `distance < 0.5` yeterlidir.

    def reset(self, target_position, target_translation): # Ortamı yeni bir bölüm için sıfırlar
        
        self.target_position = target_position # Verilen hedef pozisyonu saklar
        self.target_translation = target_translation # Verilen hedef translasyonu/rotasyonu saklar


        current_state = self.get_observation(target_position, target_translation) # Yeni hedeflerle mevcut durumu/gözlemi alır
        return current_state # Sıfırlanmış durumdaki gözlemi döndürür

    def step(self, action): # Ajanın seçtiği bir aksiyonu ortamda uygular
        if self.current_joint_angles is None: # Eğer mevcut eklem açıları henüz alınmadıysa (başlangıç veya hata durumu)
            self.get_logger().warn("Joint açıları henüz alınmadı.") # Uyarı mesajı loglar
            return np.zeros(20), -100.0, True # Varsayılan gözlem, büyük ceza ve bölümü sonlandır durumu döndürür

        # Yeni eklem durumlarını, mevcut açılara ajanın aksiyonunu (değişim miktarını) ekleyerek hesaplar
        new_joint_states = np.array(self.current_joint_angles) + np.array(action)

        msg = JointTrajectory() # Bir JointTrajectory mesajı oluşturur
        msg.joint_names = self.joint_names # Mesaja eklem adlarını atar

        point = JointTrajectoryPoint() # Bir JointTrajectoryPoint (yörünge noktası) oluşturur
        point.positions = new_joint_states.tolist() # Hesaplanan yeni eklem pozisyonlarını yörünge noktasına atar
        point.time_from_start = Duration(sec=2, nanosec=0) # Bu yörünge noktasına ulaşmak için başlangıçtan itibaren geçecek süreyi ayarlar (2 saniye)

        msg.points.append(point) # Yörünge noktalarını mesaja ekler
        self.publisher.publish(msg) # Oluşturulan yörünge mesajını yayınlar
        self.get_logger().info(f'Yeni aksiyon gönderildi: {point.positions}') # Gönderilen aksiyon bilgisini loglar

        while True: # Robotun hedeflenen yeni eklem durumlarına ulaşmasını bekler
            rclpy.spin_once(self, timeout_sec=0.1) # ROS2 callback'lerini işlemek için kısa bir süre bekler (eklem durumlarını güncellemek için)
            if self.current_joint_angles is None: # Eğer bu sırada eklem açıları kaybolursa (beklenmedik bir durum)
                continue # Döngüye devam et
            # Mevcut eklem açıları ile hedeflenen yeni eklem durumları arasındaki hatayı hesaplar
            error = np.linalg.norm(np.array(self.current_joint_angles) - new_joint_states)
            if error < 0.01: # Eğer hata belirli bir eşik değerinin altına düşerse (robot hedefe ulaştı kabul edilir)
                break # Bekleme döngüsünden çık

        obs = self.get_observation(self.target_position, self.target_translation) # Yeni durumdaki gözlemi alır
        reward = self.compute_reward() # Yeni durum için ödülü hesaplar
        done = self.is_done() # Yeni durumda bölümün bitip bitmediğini kontrol eder

        return obs, reward, done # Gözlem, ödül ve bitiş durumunu döndürür
