import rclpy # ROS2 Python istemci kütüphanesini içe aktar
from rclpy.node import Node # ROS2 Düğüm (Node) sınıfını içe aktar
from builtin_interfaces.msg import Duration # Zaman süresi mesaj tipini içe aktar
from sensor_msgs.msg import JointState # Eklem durumları mesaj tipini içe aktar
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # Eklem yörüngesi ve yörünge noktası mesaj tiplerini içe aktar
from geometry_msgs.msg import TransformStamped # Dönüşüm (transformasyon) damgalı mesaj tipini içe aktar
import tf2_ros # ROS2 için TF2 (transformasyon kütüphanesi) modülünü içe aktar
import numpy as np # NumPy kütüphanesini np takma adıyla içe aktar
from moveit_msgs.srv import GetStateValidity
import time

class RobotEnv(Node): # Robot ortamını temsil eden sınıf, rclpy.node.Node sınıfından miras alır
    def __init__(self): # Sınıfın yapıcı (constructor) metodu
        super().__init__('robot_env') # Üst sınıfın (Node) yapıcı metodunu 'robot_env' düğüm adıyla çağırır
        self.tf_buffer = tf2_ros.Buffer() # TF2 dönüşümlerini saklamak için bir arabellek (buffer) oluşturur
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self) # TF2 dönüşümlerini dinlemek için bir dinleyici (listener) oluşturur
        
        self.publisher = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10) # Eklem yörüngesi mesajlarını yayınlamak için bir yayıncı (publisher) oluşturur
        self.subscription = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10) # Eklem durumlarını dinlemek için bir abone (subscriber) oluşturur ve callback fonksiyonunu atar
        self.current_joint_angles = None # Mevcut eklem açılarını saklamak için bir değişken (başlangıçta None)
        self.last_print_time = self.get_clock().now().seconds_nanoseconds()[0] # Yazdırma zamanını izlemek için son yazdırma zamanı

        self.joint_names = [ # Kontrol edilecek eklemlerin adlarını içeren bir liste
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", # Omuz ve dirsek eklemleri
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint" # Bilek eklemleri
        ]

        self.link_names = [ # Robotun link (bağlantı) adlarını içeren bir liste (şu anki kodda doğrudan kullanılmıyor gibi görünüyor)
            "shoulder_link", "upper_arm_link", "forearm_link", # Omuz, üst kol ve ön kol linkleri
            "wrist_1_link", "wrist_2_link", "wrist_3_link" # Bilek linkleri
        ]

        # Başlangıç pozisyonu tanımla (UR robot için güvenli bir başlangıç pozisyonu)
        self.home_position = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]

        self.target_position = None # Hedef pozisyonu saklamak için bir değişken
        self.target_translation = None # Hedef rotasyonu/translasyonu saklamak için bir değişken (quaternion bekleniyor olabilir)

        # Collision kontrolü için service client ekleyelim
        self.collision_client = self.create_client(GetStateValidity, '/check_state_validity')
        while not self.collision_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Collision kontrol servisi bekleniyor...')

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

    def teleport_to_home(self):
        """
        Robotu başlangıç pozisyonuna anında ışınlar ve tamamlandığında 5 saniye bekler.
        Tüm hız ve ivme değerlerini sıfırlar, böylece robot hareketsiz başlar.
        """
        self.get_logger().info("Robot başlangıç konumuna ışınlanıyor...")
        
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = self.home_position
        
        # Açıkça tüm hız ve ivme değerlerini sıfırlıyoruz
        point.velocities = [0.0] * len(self.joint_names)  # Tüm eklemler için hız 0
        point.accelerations = [0.0] * len(self.joint_names)  # Tüm eklemler için ivme 0
        
        # Hareketi 2 saniyede yapmak için süreyi ayarlıyoruz
        point.time_from_start = Duration(sec=1, nanosec=0)  # 2 saniye
        
        msg.points.append(point)
        
        # Önce mevcut hareketleri durduralım (boş bir yörünge göndererek)
        stop_msg = JointTrajectory()
        stop_msg.joint_names = self.joint_names
        self.publisher.publish(stop_msg)
        
        # Kısa bir bekleme sonrası yeni konum komutunu gönder
        rclpy.spin_once(self, timeout_sec=0.01)
        
        # Şimdi yeni pozisyona ışınla
        self.publisher.publish(msg)
        
        # Robotun başlangıç konumuna gerçekten gidip gitmediğini kontrol et
        timeout_start = time.time()
        while time.time() - timeout_start < 2.0:  # Maksimum 2 saniye bekle
            rclpy.spin_once(self, timeout_sec=0.1)  # ROS callback'lerini işle
            
            if self.current_joint_angles is not None:
                # Mevcut eklem açıları ile hedef açılar arasındaki farkı hesapla
                error = np.linalg.norm(np.array(self.current_joint_angles) - np.array(self.home_position))
                
                # Eğer robot hedef konuma yeterince yaklaştıysa
                if error < 0.01:
                    self.get_logger().info("Robot başlangıç konumuna ulaştı, 5 saniye bekleniyor...")
                    
                    # 5 saniye bekle
                    wait_start = time.time()
                    while time.time() - wait_start < 5.0:
                        rclpy.spin_once(self, timeout_sec=0.1)
                    
                    self.get_logger().info("5 saniyelik bekleme tamamlandı.")
                    return
        
        # Eğer robor hedef konuma ulaşamadıysa, yine de devam et
        self.get_logger().warn("Robot başlangıç konumuna ulaşma zaman aşımına uğradı!")
        
        # Hareket tamamlanmasa bile iç durumu güncelle 
        if self.current_joint_angles is not None:
            self.current_joint_angles = np.array(self.home_position)

    def reset(self, target_position, target_translation): # Ortamı yeni bir bölüm için sıfırlar
        # Robotu başlangıç konumuna ışınla
        self.teleport_to_home()
        
        # End-effector konumunu al ve yazdır
        position, rotation = self.get_end_effector_position()
        if position is not None:
            self.get_logger().info(f"End-effector konumu: {position}")
        
        self.target_position = target_position # Verilen hedef pozisyonu saklar
        self.target_translation = target_translation # Verilen hedef translasyonu/rotasyonu saklar

        current_state = self.get_observation(target_position, target_translation) # Yeni hedeflerle mevcut durumu/gözlemi alır
        return current_state # Sıfırlanmış durumdaki gözlemi döndürür

    def check_collision(self):
        """Robot kolunun çarpışma durumunda olup olmadığını kontrol eder"""
        if self.current_joint_angles is None:
            return True  # Eklem açıları alınamadıysa güvenli olarak çarpışma var kabul edelim
        
        # GetStateValidity servisi için request oluştur
        request = GetStateValidity.Request()
        
        # Robot durumunu oluştur
        request.robot_state.joint_state.name = self.joint_names
        request.robot_state.joint_state.position = self.current_joint_angles[:6].tolist()  # NumPy dizisini Python listesine dönüştür
        
        # Grup adını belirt
        request.group_name = 'ur_manipulator'
        
        # Servisi çağır ve sonucu al
        future = self.collision_client.call_async(request)
        
        # Sonucu bekle
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        
        if future.result() is not None:
            result = future.result()
            if not result.valid:
                self.get_logger().warn("Çarpışma tespit edildi!")
                return True  # Çarpışma var
            else:
                return False  # Çarpışma yok
        else:
            self.get_logger().error("Servis çağrısı başarısız oldu!")
            return True  # Hata durumunda güvenli tarafta kal

    def step(self, action): # Ajanın seçtiği bir aksiyonu ortamda uygular
        if self.current_joint_angles is None: # Eğer mevcut eklem açıları henüz alınmadıysa (başlangıç veya hata durumu)
            self.get_logger().warn("Joint açıları henüz alınmadı.") # Uyarı mesajı loglar
            return np.zeros(20), 0.0, True # Varsayılan gözlem, ödül yok ve bölümü sonlandır durumu döndürür

        # Yeni eklem durumlarını, mevcut açılara ajanın aksiyonunu (değişim miktarını) ekleyerek hesaplar
        new_joint_states = np.array(self.current_joint_angles) + np.array(action)

        msg = JointTrajectory() # Bir JointTrajectory mesajı oluşturur
        msg.joint_names = self.joint_names # Mesaja eklem adlarını atar

        point = JointTrajectoryPoint() # Bir JointTrajectoryPoint (yörünge noktası) oluşturur
        point.positions = new_joint_states.tolist() # Hesaplanan yeni eklem pozisyonlarını yörünge noktasına atar
        
        # Hareketi 2 saniyede yapmak için süreyi ayarlıyoruz
        point.time_from_start = Duration(sec=1, nanosec=0)  # 1 saniye

        msg.points.append(point) # Yörünge noktalarını mesaja ekler
        self.publisher.publish(msg) # Oluşturulan yörünge mesajını yayınlar
        self.get_logger().info(f'Yeni aksiyon gönderildi: {point.positions}') # Gönderilen aksiyon bilgisini loglar
        
        # Collision kontrolü için son kontrol zamanı
        last_collision_check = time.time()

        while True: # Robotun hedeflenen yeni eklem durumlarına ulaşmasını bekler
            current_time = self.get_clock().now().seconds_nanoseconds()[0]
            if current_time - self.last_print_time >= 5.0:  # 5 saniyede bir yazdır
                print(f'current_joint_angles: {self.current_joint_angles}') # Durumu yazdır
                self.last_print_time = current_time  # Son yazdırma zamanını güncelle
            
            # Her saniye collision kontrolü yap
            if time.time() - last_collision_check >= 1.0:
                if self.check_collision():
                    # Çarpışma varsa episode'u sonlandır, ödül vermeden (0.0)
                    self.get_logger().warn("Çarpışma tespit edildi! Episode sonlandırılıyor.")
                    return self.get_observation(self.target_position, self.target_translation), 0.0, True
                last_collision_check = time.time()
            
            rclpy.spin_once(self, timeout_sec=0.1) # ROS2 callback'lerini işlemek için kısa bir süre bekler (eklem durumlarını güncellemek için)
            if self.current_joint_angles is None: # Eğer bu sırada eklem açıları kaybolursa (beklenmedik bir durum)
                continue # Döngüye devam et
            # Mevcut eklem açıları ile hedeflenen yeni eklem durumları arasındaki hatayı hesaplar
            error = np.linalg.norm(np.array(self.current_joint_angles) - new_joint_states)
            if error < 0.01: # Eğer hata belirli bir eşik değerinin altına düşerse (robot hedefe ulaştı kabul edilir)
                break # Bekleme döngüsünden çık

        # Son bir collision kontrolü daha yap
        if self.check_collision():
            return self.get_observation(self.target_position, self.target_translation), 0.0, True

        obs = self.get_observation(self.target_position, self.target_translation) # Yeni durumdaki gözlemi alır
        reward = self.compute_reward() # Yeni durum için ödülü hesaplar
        done = self.is_done() # Yeni durumda bölümün bitip bitmediğini kontrol eder

        return obs, reward, done # Gözlem, ödül ve bitiş durumunu döndürür
