import sys
import gym
import numpy as np
import rlmodels


CONFIG = {
    'EPSILON_MIN': 0.01,
    'EPSILON_MAX': 1,
    'EPSILON_DECAY': 0.995,
    'GAMMA': 0.95,
    'BUFFER_SIZE': int(2000),
    'LEARNING_RATE': 1e-3,
    'BATCH_SIZE': 32
}


def main():
    env_id = "CartPole-v0"
    env = gym.make(env_id)

    dqn_agent = rlmodels.DDQNAgent(env.observation_space.shape,
                                   [32, 32],
                                   env.action_space.n,
                                   CONFIG)

    all_rewards = []

    for i in range(1000):
        state = env.reset()
        episode_reward = 0

        done = False
        while not done:
            env.render()

            action = dqn_agent.act(state)
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
    main()
