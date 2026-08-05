import numpy as np
import tensorflow as tf
from keras import layers, models


class DDQN:
    """
    Double Deep Q-Network baseline.

    DDQN 与 DQN 的核心区别在于 target 的计算：
    - DQN:  target = r + gamma * max_a Q_target(s', a)
    - DDQN: a* = argmax_a Q_eval(s', a)
            target = r + gamma * Q_target(s', a*)

    也就是：eval 网络负责选择动作，target 网络负责估计动作价值，
    从而缓解普通 DQN 的 Q 值过估计问题。

    epsilon 沿用当前项目语义：epsilon 表示利用概率。
    """

    def __init__(
            self,
            actions,
            num_channel,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            memory_size=500,
            batch_size=32,
            replace_target_iter=100
    ):
        self.actions = actions
        self.n_actions = len(actions)
        self.num_channel = num_channel
        self.n_features = num_channel

        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter

        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2), dtype=np.float32)
        self.memory_counter = 0

        self.model_eval = self._build_dqn()
        self.model_target = self._build_dqn()
        self.model_target.set_weights(self.model_eval.get_weights())

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.lr)

    def _build_dqn(self):
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.n_features,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.n_actions, activation='linear')
        ])
        return model

    def _format_state(self, observation):
        return np.array(observation, dtype=np.float32).reshape(1, -1)

    def get_q_values(self, observation):
        observation = self._format_state(observation)
        return self.model_eval(observation, training=False)[0].numpy()

    def choose_action(self, observation):
        if np.random.uniform() < self.epsilon:
            q_values = self.get_q_values(observation)
            max_actions = np.where(q_values == np.max(q_values))[0]
            action = np.random.choice(max_actions)
        else:
            action = np.random.choice(self.actions)
        return action

    def choose_action_lstm(self, observation, prediction=None):
        """
        兼容旧训练框架命名。
        DDQN 不使用 LSTM prediction。
        """
        return self.choose_action(observation)

    def store_transition(self, s, a, r, s_):
        s = np.array(s, dtype=np.float32).reshape(-1)
        s_ = np.array(s_, dtype=np.float32).reshape(-1)
        transition = np.hstack((s, [a, r], s_))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self, s, a, r, s_):
        self.store_transition(s, a, r, s_)

        if self.memory_counter < self.memory_size:
            return

        sample_size = min(self.batch_size, self.memory_size)
        indices = np.random.choice(self.memory_size, size=sample_size, replace=False)
        batch = self.memory[indices, :]

        states = batch[:, :self.n_features]
        actions = batch[:, self.n_features].astype(np.int32)
        rewards = batch[:, self.n_features + 1]
        next_states = batch[:, -self.n_features:]

        states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards_tf = tf.convert_to_tensor(rewards, dtype=tf.float32)
        actions_tf = tf.convert_to_tensor(actions, dtype=tf.int32)

        with tf.GradientTape() as tape:
            q_eval = self.model_eval(states_tf, training=True)
            q_eval_selected = tf.reduce_sum(
                q_eval * tf.one_hot(actions_tf, self.n_actions),
                axis=1
            )

            # Double DQN target:
            # 1. eval 网络选择 next action
            # 2. target 网络评估该 next action
            q_next_eval = self.model_eval(next_states_tf, training=False)
            next_actions = tf.argmax(q_next_eval, axis=1, output_type=tf.int32)

            q_next_target = self.model_target(next_states_tf, training=False)
            q_next_selected = tf.reduce_sum(
                q_next_target * tf.one_hot(next_actions, self.n_actions),
                axis=1
            )

            q_target = rewards_tf + self.gamma * q_next_selected

            loss = tf.keras.losses.mean_squared_error(q_target, q_eval_selected)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, self.model_eval.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model_eval.trainable_variables))

        if self.memory_counter % self.replace_target_iter == 0:
            self.model_target.set_weights(self.model_eval.get_weights())
