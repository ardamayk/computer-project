import torch
from torch.serialization import add_safe_globals
import numpy.core.multiarray as multiarray

def save_model(td3_agent, replay_buffer, filename):
    filename = filename + ".pth"
    checkpoint = {
        'actor': td3_agent.actor.state_dict(),
        'actor_target': td3_agent.actor_target.state_dict(),
        'critic': td3_agent.critic.state_dict(),
        'critic_target': td3_agent.critic_target.state_dict(),
        'replay_buffer': replay_buffer.storage
    }
    torch.save(checkpoint, filename)
    print(f'Model and replay buffer saved to {filename}')

def load_model(td3_agent, replay_buffer, filename):
    # Güvenli global eklemesini yap
    add_safe_globals([multiarray._reconstruct])
    # weights_only parametresi ile yükle
    checkpoint = torch.load(filename, weights_only=False)
    td3_agent.actor.load_state_dict(checkpoint['actor'])
    td3_agent.actor_target.load_state_dict(checkpoint['actor_target'])
    td3_agent.critic.load_state_dict(checkpoint['critic'])
    td3_agent.critic_target.load_state_dict(checkpoint['critic_target'])
    replay_buffer.storage = checkpoint['replay_buffer']
    print(f'Model and replay buffer loaded from {filename}')
