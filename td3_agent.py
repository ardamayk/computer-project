import torch # PyTorch kütüphanesini içe aktar
import torch.nn as nn # PyTorch'un sinir ağı modülünü (nn) nn takma adıyla içe aktar
import torch.optim as optim # PyTorch'un optimizasyon algoritmalarını içeren modülünü (optim) optim takma adıyla içe aktar

class TD3: # Twin Delayed Deep Deterministic Policy Gradient (TD3) algoritmasını uygulayan sınıf
    def __init__(self, state_dim, action_dim, max_action, actor, critic, actor_target, critic_target, policy_freq=2): # Sınıfın yapıcı metodu
        self.actor = actor # Ana Aktör modelini saklar
        self.actor_target = actor_target # Hedef Aktör modelini saklar (ana modelin yavaş güncellenen kopyası)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3) # Ana Aktör modelinin parametreleri için Adam optimizer

        self.critic = critic # Ana Critic modelini saklar
        self.critic_target = critic_target # Hedef Critic modelini saklar (ana modelin yavaş güncellenen kopyası)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3) # Ana Critic modelinin parametreleri için Adam optimizer

        self.max_action = max_action # Maksimum aksiyon değerini saklar
        self.policy_freq = policy_freq # Politika (Aktör) ve hedef ağların güncelleme frekansını belirler (Critic güncellemelerine göre)
        self.total_it = 0 # Toplam eğitim iterasyonu sayacını başlatır

    def select_action(self, state): # Verilen bir duruma göre aksiyon seçer (genellikle çıkarım/test sırasında kullanılır)
        state = torch.FloatTensor(state.reshape(1, -1)) # Gelen durumu (NumPy dizisi olabilir) PyTorch tensörüne dönüştürür ve (1, state_dim) şeklinde yeniden boyutlandırır
        return self.actor(state).detach().cpu().numpy().flatten() # Aktör modelinden aksiyonu alır, gradyan takibini keser (detach), CPU'ya taşır, NumPy dizisine çevirir ve tek boyutlu hale getirir (flatten)

    def train(self, replay_buffer, batch_size=100, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5): # TD3 ajanını eğitir
        self.total_it += 1 # Toplam iterasyon sayacını bir artırır

        # Replay buffer'dan rastgele bir batch deneyim örneği al
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        with torch.no_grad(): # Bu blok içindeki işlemler için gradyan hesaplaması yapılmaz (hedef değerler oluşturulurken kullanılır)
            # Hedef politika yumuşatma (Target Policy Smoothing): Hedef aksiyonlara gürültü eklenir
            noise = ( # Gürültü tensörü oluştur
                torch.randn_like(action) * policy_noise # Aksiyonlarla aynı boyutta rastgele normal dağılımlı gürültü oluştur ve policy_noise ile ölçeklendir
            ).clamp(-noise_clip, noise_clip) # Gürültüyü belirli bir aralığa (-noise_clip, noise_clip) kırp

            # Kırpılmış gürültüyü ekleyerek bir sonraki aksiyonu hesapla
            next_action = ( # Bir sonraki aksiyonu belirle
                self.actor_target(next_state) + noise # Hedef aktörden bir sonraki durumu kullanarak aksiyonu al ve oluşturulan gürültüyü ekle
            ).clamp(-self.max_action, self.max_action) # Sonucu geçerli aksiyon aralığına (-max_action, max_action) kırp

            # Hedef Q-değerlerini hesapla (Clipped Double Q-Learning)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action) # Hedef critic ağlarından bir sonraki durum ve bir sonraki aksiyon için Q-değerlerini al
            # print(f'target1: {target_Q1} target2: {target_Q2}') # Hesaplanan hedef Q1 ve Q2 değerlerini yazdır (debug amaçlı olabilir)
            target_Q = torch.min(target_Q1, target_Q2) # İki hedef Q-değerinden daha küçük olanı seçerek aşırı tahmin (overestimation) sorununu azaltır
            target_Q = reward + gamma * target_Q # Bellman denklemini kullanarak nihai hedef Q-değerini hesapla (burada (1-done) faktörü eksik olabilir, duruma göre eklenmeli)
            '''(1 - done) *''' # Orijinal koddaki yorum satırı, genellikle target_Q = reward + (1 - done) * gamma * target_Q şeklinde kullanılır
        
        # Mevcut Q-değerlerini (tahminleri) al
        current_Q1, current_Q2 = self.critic(state, action) # Ana critic ağlarından mevcut durum ve aksiyon için Q-değerlerini al
        
        # Critic kaybını (loss) hesapla: Ortalama Kare Hata (MSE) kaybı
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q) # Her iki critic ağı için kayıpları hesapla ve topla

        # Critic ağını güncelle
        self.critic_optimizer.zero_grad() # Critic optimizer'ının gradyanlarını sıfırla
        critic_loss.backward() # Kayıp üzerinden geriye doğru yayılım yaparak gradyanları hesapla
        self.critic_optimizer.step() # Optimizer ile critic ağının parametrelerini güncelle

        # Gecikmeli politika (Aktör) ve hedef ağ güncellemeleri
        if self.total_it % self.policy_freq == 0: # Belirli sayıda iterasyonda bir (policy_freq) çalışır

            # Aktör kaybını hesapla
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean() # Aktörün ürettiği aksiyonlar için Q1 değerini maksimize etmeye çalışır (negatifini minimize ederek)

            # Aktör ağını güncelle
            self.actor_optimizer.zero_grad() # Aktör optimizer'ının gradyanlarını sıfırla
            actor_loss.backward() # Kayıp üzerinden geriye doğru yayılım yaparak gradyanları hesapla
            self.actor_optimizer.step() # Optimizer ile aktör ağının parametrelerini güncelle

            # Hedef ağları yavaşça güncelle (soft update)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()): # Ana critic ve hedef critic parametreleri üzerinde döngü
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data) # Hedef parametreleri ana parametrelere doğru küçük bir adımla (tau) yaklaştır

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()): # Ana aktör ve hedef aktör parametreleri üzerinde döngü
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data) # Hedef parametreleri ana parametrelere doğru küçük bir adımla (tau) yaklaştır
