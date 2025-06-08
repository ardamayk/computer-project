import os
import torch
import pandas as pd
import numpy as np

def save_model(td3_agent, replay_buffer, model_name):
    """Model ve replay buffer'ı kaydeder"""
    try:
        # Model klasörünü oluştur
        model_dir = os.path.join("checkpoints", model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Model dosyasını kaydet
        model_path = os.path.join(model_dir, "model.pth")
        checkpoint = {
            'actor_state_dict': td3_agent.actor.state_dict(),
            'critic_state_dict': td3_agent.critic.state_dict(),
            'actor_target_state_dict': td3_agent.actor_target.state_dict(),
            'critic_target_state_dict': td3_agent.critic_target.state_dict(),
            'actor_optimizer_state_dict': td3_agent.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': td3_agent.critic_optimizer.state_dict(),
            'replay_buffer': replay_buffer.storage
        }
        torch.save(checkpoint, model_path)
        print(f"📦 Model kaydedildi: {model_path}")
    except Exception as e:
        print(f"❌ Model kaydetme hatası: {e}")
        raise

def load_model(td3_agent, replay_buffer, model_name):
    """Model ve replay buffer'ı yükler"""
    model_path = os.path.join("checkpoints", model_name, "model.pth")
    if not os.path.exists(model_path):
        print(f"❌ Hata: Model dosyası bulunamadı: {model_path}")
        return False
        
    checkpoint = torch.load(model_path, weights_only=False)
    
    td3_agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    td3_agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    td3_agent.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
    td3_agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
    td3_agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    td3_agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    
    replay_buffer.storage = checkpoint['replay_buffer']
    print(f"✅ Model yüklendi: {model_path}")
    return True

def save_rewards(rewards, model_name):
    """Reward değerlerini episode numarası ile birlikte kaydeder. Önceki reward'lar varsa, yeni reward'lar en son episode'dan devam eder."""
    try:
        model_dir = os.path.join("checkpoints", model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        rewards_path = os.path.join(model_dir, "rewards.csv")

        # Önceki reward'ları oku (varsa)
        if os.path.exists(rewards_path):
            df_prev = pd.read_csv(rewards_path)
            last_episode = df_prev['episode'].max() if not df_prev.empty else -1
        else:
            df_prev = pd.DataFrame(columns=['episode', 'reward'])
            last_episode = -1

        # Yeni reward'lar için episode numarası oluştur
        new_episodes = np.arange(last_episode + 1, last_episode + 1 + len(rewards))
        df_new = pd.DataFrame({'episode': new_episodes, 'reward': rewards})

        # Önceki ve yeni reward'ları birleştir
        df_all = pd.concat([df_prev, df_new], ignore_index=True)
        df_all.to_csv(rewards_path, index=False)
        print(f"📊 Reward değerleri kaydedildi: {rewards_path}")
    except Exception as e:
        print(f"❌ Reward kaydetme hatası: {e}")
        raise

def load_rewards(model_name):
    """Belirtilen model klasöründen reward değerlerini episode numarası ile birlikte yükler."""
    rewards_path = os.path.join("checkpoints", model_name, "rewards.csv")
    if not os.path.exists(rewards_path):
        print(f"❌ Hata: Reward dosyası bulunamadı: {rewards_path}")
        return None, None
    df = pd.read_csv(rewards_path)
    print(f"✅ Reward değerleri yüklendi: {rewards_path}")
    return df['episode'].values, df['reward'].values

def save_actor_losses(model_dir, episodes, losses):
    """Actor loss değerlerini CSV dosyasına kaydeder."""
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        df = pd.DataFrame({
            'episode': episodes,
            'actor_loss': losses
        })
        df.to_csv(os.path.join(model_dir, "actor_losses.csv"), index=False)
        print(f"  📊 Actor loss değerleri kaydedildi: {model_dir}")
        return True
    except Exception as e:
        print(f"❌ Actor loss kaydetme hatası: {e}")
        return False

def save_critic1_losses(model_dir, episodes, losses):
    """Critic1 loss değerlerini CSV dosyasına kaydeder."""
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        df = pd.DataFrame({
            'episode': episodes,
            'critic1_loss': losses
        })
        df.to_csv(os.path.join(model_dir, "critic1_losses.csv"), index=False)
        print(f"  📊 Critic1 loss değerleri kaydedildi: {model_dir}")
        return True
    except Exception as e:
        print(f"❌ Critic1 loss kaydetme hatası: {e}")
        return False

def save_critic2_losses(model_dir, episodes, losses):
    """Critic2 loss değerlerini CSV dosyasına kaydeder."""
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        df = pd.DataFrame({
            'episode': episodes,
            'critic2_loss': losses
        })
        df.to_csv(os.path.join(model_dir, "critic2_losses.csv"), index=False)
        print(f"  📊 Critic2 loss değerleri kaydedildi: {model_dir}")
        return True
    except Exception as e:
        print(f"❌ Critic2 loss kaydetme hatası: {e}")
        return False

def save_checkpoint(td3_agent, replay_buffer, rewards, model_name, 
                   combined_episodes=None, combined_actor_losses=None, 
                   combined_critic1_losses=None, combined_critic2_losses=None):
    """Model, replay buffer, reward ve loss değerlerini kaydeder."""
    try:
        # Model ve reward'ları kaydet
        save_model(td3_agent, replay_buffer, model_name)
        save_rewards(rewards, model_name)
        
        # Loss değerlerini kaydet (eğer verilmişse)
        model_dir = os.path.join("checkpoints", model_name)
        if all(x is not None for x in [combined_episodes, combined_actor_losses, 
                                     combined_critic1_losses, combined_critic2_losses]):
            save_actor_losses(model_dir, combined_episodes, combined_actor_losses)
            save_critic1_losses(model_dir, combined_episodes, combined_critic1_losses)
            save_critic2_losses(model_dir, combined_episodes, combined_critic2_losses)
        
        print(f"✅ Checkpoint kaydedildi: {model_name}")
        return True
    except Exception as e:
        print(f"❌ Checkpoint kaydetme hatası: {e}")
        return False

def load_checkpoint(td3_agent, replay_buffer, model_name):
    """Model, replay buffer ve reward'ları yükler."""
    if not load_model(td3_agent, replay_buffer, model_name):
        return False, (None, None)
    episodes, rewards = load_rewards(model_name)
    return True, (episodes, rewards)

def list_available_models():
    """Mevcut modelleri listeler."""
    checkpoints_dir = 'checkpoints'
    if not os.path.exists(checkpoints_dir):
        return []
    return [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))] 