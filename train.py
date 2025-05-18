import rclpy
import numpy as np
import threading
from env import RobotEnv
from td3 import Actor, Critic, ReplayBuffer, TD3
from rclpy.executors import MultiThreadedExecutor

def main():
    rclpy.init()
    env = RobotEnv()

    # Start a multi-threaded executor in a separate thread
    executor = MultiThreadedExecutor()
    executor.add_node(env)
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    state_dim = 20
    action_dim = 6
    max_action = 0.2

    actor = Actor(state_dim, action_dim, max_action)
    actor_target = Actor(state_dim, action_dim, max_action)
    actor_target.load_state_dict(actor.state_dict())

    critic = Critic(state_dim, action_dim)
    critic_target = Critic(state_dim, action_dim)
    critic_target.load_state_dict(critic.state_dict())

    replay_buffer = ReplayBuffer()

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

    episodes = 1000
    best_reward = float('inf')  # Çünkü tüm reward'lar negatif

    for ep in range(episodes):
        target_position = np.array([0.5, 0.2, 1])
        target_translation = [0, 0, 0, 0]

        print(f'\n--- Episode {ep+1} ---')
        print(f'Target Position: {target_position}')
        print(f'Target Translation: {target_translation}')

        state = env.reset(target_position, target_translation)
        print(f'Initial State: {state}')

        done = False
        total_reward = 0

        while not done:
            action = td3_agent.select_action(np.array(state))
            print(f'Action: {action}')

            next_state, reward, done = env.step(action)
            print(f'Next State: {next_state}')
            print(f'Reward: {reward}, Done: {done}')

            replay_buffer.add(state, action, reward, next_state, float(done))

            if replay_buffer.size() > 25:
                td3_agent.train(replay_buffer)

            state = next_state
            total_reward += reward

        print(f'Episode {ep+1} Total Reward: {total_reward:.2f}')

        if total_reward < best_reward:
            best_reward = total_reward
            print(f'✅ New Best Reward: {best_reward:.2f}')

    # Shutdown
    env.destroy_node()
    rclpy.shutdown()
    # executor_thread.join() kaldırıldı

if __name__ == '__main__':
    main()
