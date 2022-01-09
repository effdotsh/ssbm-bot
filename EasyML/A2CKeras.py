import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0


class A2CAgent:
    def __init__(self, num_inputs, num_actions, discount_factor=0.99, learning_rate=0.01):
        self.num_actions = num_actions
        self.num_inputs = num_inputs
        self.discount_factor = discount_factor
        self.model = self.generate_model(num_inputs=num_inputs, num_actions=num_actions)

        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.huber_loss = keras.losses.Huber()
        self.action_probs_history = []
        self.critic_value_history = []
        self.rewards_history = []
        self.running_reward = 0

        self.episode_reward = 0
        self.episode_count = 0


    def generate_model(self, num_inputs, num_actions):
        inputs = layers.Input(shape=(num_inputs,))
        common = layers.Dense(128, activation="relu")(inputs)
        action = layers.Dense(num_actions, activation="softmax")(common)
        critic = layers.Dense(1)(common)

        model = keras.Model(inputs=inputs, outputs=[action, critic])
        return model

    def predict(self, state):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        # Predict action probabilities and estimated future rewards
        # from environment state
        action_probs, critic_value = self.model(state)
        self.critic_value_history.append(critic_value[0, 0])

        # Sample action from action probability distribution
        action = np.random.choice(self.num_actions, p=np.squeeze(action_probs))
        self.action_probs_history.append(tf.math.log(action_probs[0, action]))
        return action

    def update_replay_buffer(self, transition):
        self.rewards_history.append(reward)
        self.episode_reward += reward

    def train(self):
        with tf.GradientTape() as tape:

            self.running_reward = 0.05 * self.episode_reward + (1 - 0.05) * self.running_reward

            # Calculate expected value from rewards
            # - At each timestep what was the total reward received after that timestep
            # - Rewards in the past are discounted by multiplying them with gamma
            # - These are the labels for our critic
            returns = []
            discounted_sum = 0
            for r in self.rewards_history[::-1]:
                discounted_sum = r + self.discount_factor * discounted_sum
                returns.insert(0, discounted_sum)

            # Normalize
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            returns = returns.tolist()

            # Calculating loss values to update our network
            history = zip(self.action_probs_history, self.critic_value_history, returns)
            actor_losses = []
            critic_losses = []
            for log_prob, value, ret in history:
                # At this point in history, the critic estimated that we would get a
                # total reward = `value` in the future. We took an action with log probability
                # of `log_prob` and ended up recieving a total reward = `ret`.
                # The actor must be updated so that it predicts an action that leads to
                # high rewards (compared to critic's estimate) with high probability.
                diff = ret - value
                actor_losses.append(-log_prob * diff)  # actor loss

                # The critic must be updated so that it predicts a better estimate of
                # the future rewards.
                critic_losses.append(
                    self.huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                )

            # Backpropagation
            # print(self.model.trainable_variables)
            # wedewf
            loss_value = sum(actor_losses) + sum(critic_losses)
            grads = tape.gradient(loss_value, self.model.trainable_variables)

            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            # Clear the loss and reward history
            self.action_probs_history.clear()
            self.critic_value_history.clear()
            self.rewards_history.clear()

            # Log details
            self.episode_count += 1
            self.episode_reward = 0