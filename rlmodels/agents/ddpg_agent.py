import tensorflow as tf
from tensorflow import keras
import numpy as np
from ..memories import PrioritizedMemory
from ..utils import OUNoise


class Actor(keras.Model):
    def __init__(self, state_size, action_size, units):
        super().__init__()

        self.hidden_layers = []
        for unit in units:
            if self.hidden_layers:
                self.hidden_layers.append(keras.layers.Dense(unit, activation='relu'))
            else:
                self.hidden_layers.append(keras.layers.Dense(
                    unit, input_shape=(state_size,), activation='relu'))

        self.hidden_layers.append(keras.layers.Dense(action_size, activation='sigmoid'))

    def call(self, states):
        x = states
        for layer in self.hidden_layers:
            x = layer(x)

        return x


class Critic(keras.Model):
    def __init__(self, state_size, action_size, units):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.hidden_layers = []
        for unit in units:
            if self.hidden_layers:
                self.hidden_layers.append(keras.layers.Dense(unit, activation='relu'))
            else:
                self.hidden_layers.append(keras.layers.Dense(unit, input_shape=(
                    state_size+action_size,), activation='relu'))

        self.hidden_layers.append(keras.layers.Dense(1))

    def call(self, states, actions):
        x = tf.concat([states, actions], axis=-1)
        for layer in self.hidden_layers:
            x = layer(x)

        return x


BATCH_SIZE = 512
GAMMA = 0.99
TAU = 0.01
TRAIN_EVERY = 4


class DDPGAgent():
    def __init__(self, state_size, action_size, action_limits, config, seed=None):
        self.state_size = state_size
        self.action_size = action_size

        self.actor_local = Actor(state_size, action_size, config['actor_units'])
        self.actor_target = Actor(state_size, action_size, config['actor_units'])

        self.critic_local = Critic(state_size, action_size, config['critic_units'])
        self.critic_target = Critic(state_size, action_size, config['critic_units'])

        self.action_high = action_limits[0]
        self.action_low = action_limits[1]

        self.mse = keras.losses.MeanSquaredError()
        self.optimizer = keras.optimizers.Adam()

        self.noise = OUNoise(action_size, seed)
        self.memory = PrioritizedMemory(int(config['memory_size']), seed)

        self.global_step = 0

    def act(self, state, noisy=True):
        with tf.device('/cpu:0'):
            state_tensor = tf.constant(state, shape=(1, self.state_size), dtype=tf.float32)
            action = self.actor_local(state_tensor)
            action = keras.backend.eval(action).squeeze(axis=0)

        if noisy:
            action += self.noise.sample()

        return np.clip(action, self.action_low, self.action_high)

    # def step(self, state, action, reward, next_state, done):
    #     self.memory.add((state, action, reward, next_state, done))

    #     self.global_step += 1
    #     if self.global_step % TRAIN_EVERY == 0:
    #         experiences = self._sample(BATCH_SIZE)
    #         self.learn(experiences)

    def _error(self, state, action, reward, next_state, done):
        with tf.device('/cpu:0'):
            # Q_expected
            state_tensor = tf.constant(state, shape=(1, self.state_size), dtype=tf.float32)
            action_tensor = tf.constant(action, shape=(1, self.action_size), dtype=tf.float32)
            Q_expected = self.critic_local(state_tensor, action_tensor)

            next_state_tensor = tf.constant(next_state, shape=(1, self.state_size),
                                            dtype=tf.float32)
            next_action = self.actor_local(next_state_tensor)
            Q_target_next = self.critic_target(next_state_tensor, next_action)

            Q_expected = keras.backend.eval(Q_expected).squeeze()
            Q_target_next = keras.backend.eval(Q_target_next).squeeze()

        Q_target = reward + (GAMMA * Q_target_next * (1 - float(done)))
        error = np.abs(Q_expected - Q_target)
        return error

    def step(self, state, action, reward, next_state, done):
        # Compute error
        error = self._error(state, action, reward, next_state, done)
        self.memory.add(error, (state, action, reward, next_state, done))

        self.global_step += 1
        if self.global_step % TRAIN_EVERY == 0:
            batch = self._sample(BATCH_SIZE)
            self.learn(batch)

    def _sample(self, n):
        experiences, idxs, weights = self.memory.sample(n)

        states, actions, rewards, next_states, dones = zip(*experiences)

        states = tf.constant(states, shape=(n, self.state_size), dtype=tf.float32)
        actions = tf.constant(actions, shape=(n, self.action_size), dtype=tf.float32)
        rewards = tf.constant(rewards, shape=(n, 1), dtype=tf.float32)
        next_states = tf.constant(next_states, shape=(n, self.state_size), dtype=tf.float32)
        dones = tf.constant(dones, shape=(n, 1), dtype=tf.float32)

        weights = tf.constant(weights, shape=(n, 1), dtype=tf.float32)

        return (states, actions, rewards, next_states, dones), idxs, weights

    def learn(self, experiences):
        # With priority sampling.
        experiences, idxs, weights = experiences
        states, actions, rewards, next_states, dones = experiences

        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))

        with tf.GradientTape(persistent=True) as tape:
            Q_expected = self.critic_local(states, actions)
            critic_loss = self.mse(Q_expected, Q_targets, sample_weight=weights)

            actions_pred = self.actor_local(states)
            actor_loss = -tf.reduce_mean(self.critic_local(states, actions_pred) * weights)

            loss = critic_loss + actor_loss

        # Update the memory.
        errors = np.abs(keras.backend.eval(Q_expected - Q_targets))

        for i, idx in enumerate(idxs):
            self.memory.update(idx, errors[i])

        critic_grads = tape.gradient(loss, self.critic_local.trainable_variables)
        actor_grads = tape.gradient(loss, self.actor_local.trainable_variables)
        del tape

        self.optimizer.apply_gradients(zip(critic_grads, self.critic_local.trainable_variables))
        self.optimizer.apply_gradients(zip(actor_grads, self.actor_local.trainable_variables))

        self._update_targets()

    def _update_targets(self):
        actor_target_weights = self.actor_target.get_weights()
        actor_local_weights = self.actor_local.get_weights()
        self.actor_target.set_weights([(1-TAU)*target_weights + TAU*local_weights for target_weights,
                                       local_weights in zip(actor_target_weights, actor_local_weights)])

        critic_target_weights = self.critic_target.get_weights()
        critic_local_weights = self.critic_local.get_weights()
        self.critic_target.set_weights([(1-TAU)*target_weights + TAU*local_weights for target_weights,
                                        local_weights in zip(critic_target_weights, critic_local_weights)])

    def reset(self):
        self.noise.reset()
