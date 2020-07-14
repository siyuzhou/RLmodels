import os
import sys
import gym
import numpy as np
import rlmodels


def main():
    env_id = "MountainCar-v0"
    env = gym.make(env_id)

    dqn_agent = rlmodels.ActorCriticAgent(env.observation_space.shape,
                                          env.action_space.n,
                                          [32, 32])

    all_rewards = []

    for i in range(1000):
        state = env.reset()
        episode_reward = 0

        done = False
        while not done:
            # env.render()

            action, prob = dqn_agent.act(state)
            next_state, reward, done, _ = env.step(action)
            dqn_agent.step(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

        all_rewards.append(episode_reward)

        print(
            f'\rEpisode {i} reward: {episode_reward}.  Average: {np.mean(all_rewards[-200:]):.2f}', end='')
        sys.stdout.flush()

        if (i+1) % 10 == 0:
            print('')

    env.close()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    main()
