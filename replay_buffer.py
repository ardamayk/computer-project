import numpy as np
import torch

class ReplayBuffer: # Deneyim tekrarı (Replay Buffer) mekanizmasını uygulayan sınıf
    def __init__(self, max_size=int(1e6)): # Sınıfın yapıcı metodu
        self.storage = [] # Deneyimleri (geçişleri) saklamak için bir liste
        self.max_size = max_size # Buffer'ın maksimum kapasitesi
        self.ptr = 0 # Buffer'da bir sonraki deneyimin yazılacağı pozisyonu gösteren işaretçi

    def add(self, state, action, reward, next_state, done): # Buffer'a yeni bir deneyim ekler
        data = (state, action, reward, next_state, done) # Deneyimi bir demet (tuple) olarak paketler
        if len(self.storage) < self.max_size: # Eğer buffer dolu değilse
            self.storage.append(data) # Deneyimi doğrudan listeye ekler
        else: # Eğer buffer doluysa (eski deneyimlerin üzerine yazılır)
            self.storage[self.ptr] = data # İşaretçinin gösterdiği konuma yeni deneyimi yazar
            self.ptr = (self.ptr + 1) % self.max_size # İşaretçiyi dairesel (circular) olarak bir sonraki pozisyona günceller

    def sample(self, batch_size=20): # Buffer'dan rastgele bir mini-batch (küçük parti) deneyim örneği alır
        ind = np.random.randint(0, len(self.storage), size=batch_size) # Buffer boyutundan rastgele indeksler seçer
        batch = [self.storage[i] for i in ind] # Seçilen indekslere karşılık gelen deneyimleri bir liste olarak toplar

        state, action, reward, next_state, done = map(np.stack, zip(*batch)) # Toplanan batch'teki deneyimleri ayrıştırır ve her bir bileşeni (state, action vb.) NumPy dizileri haline getirir
        return ( # Ayrıştırılmış ve NumPy dizilerine dönüştürülmüş deneyimleri PyTorch tensörleri olarak döndürür
            torch.FloatTensor(state), # Durumları içeren tensör
            torch.FloatTensor(action), # Aksiyonları içeren tensör
            torch.FloatTensor(reward).unsqueeze(1), # Ödülleri içeren tensör (tek boyutlu sütun vektörü yapmak için unsqueeze(1))
            torch.FloatTensor(next_state), # Sonraki durumları içeren tensör
            torch.FloatTensor(done).unsqueeze(1) # Bitiş (done) bayraklarını içeren tensör (tek boyutlu sütun vektörü yapmak için unsqueeze(1))
        )

    def size(self): # Buffer'da mevcut olan deneyim sayısını döndürür
        return len(self.storage) # Saklama listesinin uzunluğunu döndürür