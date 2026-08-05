import numpy as np
import tensorflow as tf
from keras import layers, models


class DRQN:
    """
    Deep Recurrent Q-Network (DRQN)

    与当前 DQNLSTM.py 的区别：
    - DQNLSTM.py 是“DQN + 单独 LSTM 预测模块”；
    - 本文件中的 DRQN 是“LSTM 直接作为 Q 网络的一部分”，
      输入一段历史状态序列，输出每个信道动作的 Q 值。

    在本项目中：
    - state/action 对应动态频谱接入中的频谱观测状态/信道选择动作；
    - epsilon 仍沿用项目原有语义：epsilon 表示利用概率。
      即 random < epsilon 时选择最大 Q 值动作，否则随机探索。
    """

    def __init__(
            self,
            actions,
            num_channel,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            sequence_length=10,
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

        self.sequence_length = sequence_length
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.replace_target_iter = replace_target_iter

        # 每条经验包含：
        # state_sequence + action + reward + next_state_sequence
        self.memory_width = self.sequence_length * self.n_features * 2 + 2
        self.memory = np.zeros((self.memory_size, self.memory_width), dtype=np.float32)
        self.memory_counter = 0

        # 保存在线观测历史，用于构造 DRQN 输入序列
        self.state_history = []

        self.model_eval = self._build_drqn()
        self.model_target = self._build_drqn()
        self.model_target.set_weights(self.model_eval.get_weights())

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.lr)

    def _build_drqn(self):
        model = models.Sequential([
            layers.Input(shape=(self.sequence_length, self.n_features)),
            layers.LSTM(50, activation='sigmoid'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.n_actions, activation='linear')
        ])
        return model

    def _format_state(self, observation):
        return np.array(observation, dtype=np.float32).reshape(-1)

    def _make_sequence_from_history(self, observation):
        """
        使用历史状态 + 当前状态构造固定长度序列。
        历史不足 sequence_length 时，用 0 填充。
        """
        current_state = self._format_state(observation)
        history = self.state_history + [current_state]
        history = history[-self.sequence_length:]

        if len(history) < self.sequence_length:
            padding = [np.zeros(self.n_features, dtype=np.float32)
                       for _ in range(self.sequence_length - len(history))]
            history = padding + history

        return np.array(history, dtype=np.float32)

    def _to_sequence(self, state):
        """
        兼容两种输入：
        - 若 state 是一维状态，则构造为长度 sequence_length 的序列；
        - 若 state 已经是二维序列，则直接调整为固定格式。
        """
        state = np.array(state, dtype=np.float32)

        if state.ndim == 1:
            return self._make_sequence_from_history(state)

        if state.ndim == 2:
            if state.shape[0] >= self.sequence_length:
                return state[-self.sequence_length:, :]
            padding = np.zeros(
                (self.sequence_length - state.shape[0], self.n_features),
                dtype=np.float32
            )
            return np.vstack((padding, state))

        raise ValueError("state must be a 1-D observation or a 2-D state sequence")

    def record_state(self, observation):
        """
        记录当前观测状态，供后续构造历史序列。
        """
        state = self._format_state(observation)
        self.state_history.append(state)
        if len(self.state_history) > self.sequence_length:
            self.state_history = self.state_history[-self.sequence_length:]

    def channel_record(self, channel_state, access_act=None, action=None, su_index=None, num_su=None):
        """
        为了兼容项目中已有 LSTM 方法的调用接口。

        DRQN 不再单独训练一个 LSTM 预测器，因此这里只记录 SU 可观测到的信道状态，
        用作 recurrent Q-network 的历史输入。
        """
        self.record_state(channel_state)

    def get_q_values(self, state):
        state_sequence = self._to_sequence(state)
        state_tensor = tf.convert_to_tensor(
            np.expand_dims(state_sequence, axis=0),
            dtype=tf.float32
        )
        return self.model_eval(state_tensor, training=False)[0].numpy()

    def choose_action(self, observation):
        """
        epsilon 概率利用：选择 DRQN 输出 Q 值最大的信道；
        1 - epsilon 概率探索：随机选择信道。
        """
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
        DRQN 的 LSTM 已经在 Q 网络内部，因此不需要外部 prediction。
        """
        return self.choose_action(observation)

    def store_transition(self, s, a, r, s_):
        state_sequence = self._to_sequence(s)
        next_state_sequence = self._to_sequence(s_)

        transition = np.hstack((
            state_sequence.reshape(-1),
            [a, r],
            next_state_sequence.reshape(-1)
        ))

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

        state_end = self.sequence_length * self.n_features
        action_index = state_end
        reward_index = state_end + 1
        next_state_start = state_end + 2

        states = batch[:, :state_end].reshape(
            sample_size,
            self.sequence_length,
            self.n_features
        )
        actions = batch[:, action_index].astype(np.int32)
        rewards = batch[:, reward_index]
        next_states = batch[:, next_state_start:].reshape(
            sample_size,
            self.sequence_length,
            self.n_features
        )

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

            q_next = self.model_target(next_states_tf, training=False)
            q_target = rewards_tf + self.gamma * tf.reduce_max(q_next, axis=1)

            loss = tf.keras.losses.mean_squared_error(q_target, q_eval_selected)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, self.model_eval.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model_eval.trainable_variables))

        if self.memory_counter % self.replace_target_iter == 0:
            self.model_target.set_weights(self.model_eval.get_weights())
