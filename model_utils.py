import torch # PyTorch kütüphanesini içe aktar
import torch.nn as nn # PyTorch'un sinir ağı modülünü (nn) nn takma adıyla içe aktar

class Actor(nn.Module): # Aktör (Policy) modelini tanımlayan sınıf, PyTorch'un nn.Module sınıfından miras alır
    def __init__(self, state_dim, action_dim, max_action): # Sınıfın yapıcı (constructor) metodu
        super(Actor, self).__init__() # Miras aldığı nn.Module sınıfının yapıcı metodunu çağır
        self.layer = nn.Sequential( # Sıralı bir sinir ağı katmanı oluşturur
            nn.Linear(state_dim, 400), # Durum boyutundan 400 nörona tam bağlı (doğrusal) katman
            nn.ReLU(), # ReLU (Rectified Linear Unit) aktivasyon fonksiyonu
            nn.Linear(400, 300), # 400 nörondan 300 nörona tam bağlı katman
            nn.ReLU(), # ReLU aktivasyon fonksiyonu
            nn.Linear(300, action_dim), # 300 nörondan aksiyon boyutuna tam bağlı katman
            nn.Tanh() # Tanh (hiperbolik tanjant) aktivasyon fonksiyonu (çıktıyı -1 ile 1 arasına sıkıştırır)
        )
        self.max_action = max_action # Aksiyonun maksimum değerini saklar (çıktıyı ölçeklendirmek için)

    def forward(self, state): # Modelin ileri yayılım (forward pass) metodunu tanımlar
        return self.max_action * self.layer(state) # Durumu sinir ağı katmanından geçirir ve sonucu max_action ile ölçeklendirir

class Critic(nn.Module): # Critic (Value) modelini tanımlayan sınıf, PyTorch'un nn.Module sınıfından miras alır
    def __init__(self, state_dim, action_dim): # Sınıfın yapıcı metodu
        super(Critic, self).__init__() # Miras aldığı nn.Module sınıfının yapıcı metodunu çağır
        # Twin Delayed DDPG (TD3) için iki ayrı Q-fonksiyonu (Critic) ağı tanımlanır (Q1 ve Q2)
        self.q1 = nn.Sequential( # Birinci Q-ağı (q1)
            nn.Linear(state_dim + action_dim, 400), # Durum ve aksiyon boyutunun toplamından 400 nörona tam bağlı katman
            nn.ReLU(), # ReLU aktivasyon fonksiyonu
            nn.Linear(400, 300), # 400 nörondan 300 nörona tam bağlı katman
            nn.ReLU(), # ReLU aktivasyon fonksiyonu
            nn.Linear(300, 1) # 300 nörondan tek bir Q-değerine (skaler) tam bağlı katman
        )
        self.q2 = nn.Sequential( # İkinci Q-ağı (q2)
            nn.Linear(state_dim + action_dim, 400), # Durum ve aksiyon boyutunun toplamından 400 nörona tam bağlı katman
            nn.ReLU(), # ReLU aktivasyon fonksiyonu
            nn.Linear(400, 300), # 400 nörondan 300 nörona tam bağlı katman
            nn.ReLU(), # ReLU aktivasyon fonksiyonu
            nn.Linear(300, 1) # 300 nörondan tek bir Q-değerine (skaler) tam bağlı katman
        )

    def forward(self, state, action): # Modelin ileri yayılım metodunu tanımlar
        sa = torch.cat([state, action], 1) # Durum ve aksiyon tensörlerini birleştirir (concatenate)
        return self.q1(sa), self.q2(sa) # Birleştirilmiş durum-aksiyon çiftini her iki Q-ağından geçirir ve sonuçları döndürür

    def q1_forward(self, state, action): # Sadece birinci Q-ağının (q1) ileri yayılımını yapar
        sa = torch.cat([state, action], 1) # Durum ve aksiyon tensörlerini birleştirir
        return self.q1(sa) # Birleştirilmiş durum-aksiyon çiftini q1 ağından geçirir ve sonucu döndürür