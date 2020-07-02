import os
import sys
import gym
import numpy as np
import rlmodels
from rlmodels.utils import Config


CHECKPOINT = 'checkpoint_pendulum'


def main():
    env_id = "Pendulum-v0"
    env = gym.make(env_id)
    config = Config(memory_capacity=int(1e5),
                    action_high=env.action_space.high,
                    action_low=env.action_space.low)

    agent = rlmodels.DDPGAgent(env.observation_space.shape[0],
                               env.action_space.shape[0],
                               [32, 32],
                               [32, 32],
                               config=config)

    agent.load_network(CHECKPOINT)

    all_rewards = []

    for i in range(10000):
        state = env.reset()
        episode_reward = 0

        done = False
        t = 0
        t_max = 300
        while (not done) and (t < t_max):
            env.render()

            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            t += 1

        all_rewards.append(episode_reward)

        print(
            f'\rEpisode {i} reward: {episode_reward:.2f}.  Average: {np.mean(all_rewards[-200:]):.2f}', end='')
        sys.stdout.flush()

        if (i+1) % 10 == 0:
            agent.save_network(CHECKPOINT)
            print('')

    env.close()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    main()
