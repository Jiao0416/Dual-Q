"""
文件名：DualQeRandom.py
功能：在双Q表 + LSTM 后验融合框架下，使用 ε-random 探索策略。

核心对比点：
1. 利用阶段：根据 LSTM 加权融合后的双Q值选择最大动作；
2. 探索阶段：均匀随机选择信道。

该文件用于作为 DualQ_plus.py 中 LSTM-directed exploration 的经典 ε-greedy baseline。
"""

import numpy as np
import pandas as pd
from keras import layers, models
import tensorflow as tf


pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


class DualQERandom:
    def __init__(
            self,
            actions,
            num_channel,
            learning_rate=0.5,
            reward_decay=0.9,
            e_greedy=0.9
    ):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        # === 双Q表设计 ===
        self.q_table_access = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.q_table_conflict = pd.DataFrame(columns=self.actions, dtype=np.float64)

        self.num_channel = num_channel
        self.channel_data = []

        # LSTM 模型与优化器
        self._build_lstm()
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    def _build_lstm(self):
        """构建LSTM预测模型"""
        self.model = models.Sequential()
        self.model.add(layers.LSTM(units=50, activation='relu', input_shape=(10, self.num_channel)))
        self.model.add(layers.Dense(units=self.num_channel, activation='sigmoid'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Reshape((1, self.num_channel)))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    def channel_record(self, channel_state, access_act, action, su_index, num_su):
        """记录信道状态：PU占用 + 其他SU成功接入占用"""
        channel_data_current = np.zeros(self.num_channel).astype(np.int32)

        for i in range(self.num_channel):
            channel_data_current[i] = channel_state[i]

        for i in range(num_su):
            if access_act[i] == 1 and i != su_index:
                channel_data_current[action[i]] = 1

        self.channel_data.append(channel_data_current)

    def channel_prediction(self, step):
        """LSTM在线训练与预测"""
        if len(self.channel_data) < 11:
            return np.zeros((1, self.num_channel))

        x_train = np.expand_dims(self.channel_data[step - 11: step - 1], axis=0)
        y_train = np.expand_dims(self.channel_data[step - 1: step], axis=0)

        with tf.GradientTape() as tape:
            y_pred = self.model(x_train, training=True)
            loss = tf.keras.losses.mean_squared_error(y_train, y_pred)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        input_sequence = np.expand_dims(self.channel_data[step - 10: step], axis=0)
        prediction = np.squeeze(self.model(input_sequence, training=False), axis=0)
        return prediction

    def check_state_exist(self, state):
        """检查状态是否存在，不存在则添加"""
        if state not in self.q_table_access.index:
            new_row = pd.Series([0] * len(self.actions), index=self.q_table_access.columns, name=state)
            self.q_table_access = self.q_table_access._append(new_row)

        if state not in self.q_table_conflict.index:
            new_row = pd.Series([0] * len(self.actions), index=self.q_table_conflict.columns, name=state)
            self.q_table_conflict = self.q_table_conflict._append(new_row)

    def _get_weighted_q_values(self, observation, prediction, num_channel):
        """根据 LSTM 预测概率融合 Q_access 和 Q_conflict"""
        q_values_access = self.q_table_access.loc[observation, :]
        q_values_conflict = self.q_table_conflict.loc[observation, :]

        ones = np.ones(num_channel)
        weighted_q_values = (
            q_values_access * (ones - prediction[0]) +
            q_values_conflict * prediction[0]
        )

        return weighted_q_values.astype(np.float64)

    def choose_action_lstm(self, observation, prediction, num_channel):
        """
        ε-random 信道选择策略：
        - np.random.uniform() < epsilon：利用，选择 fused Q 最大动作；
        - 否则：探索，均匀随机选择动作。
        """
        self.check_state_exist(observation)
        weighted_q_values = self._get_weighted_q_values(observation, prediction, num_channel)

        if np.random.uniform() < self.epsilon:
            weighted_q_values = weighted_q_values.reindex(np.random.permutation(weighted_q_values.index))
            action = weighted_q_values.idxmax()
        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_, r_type):
        """双Q表更新：成功更新 Q_access，碰撞更新 Q_conflict"""
        self.check_state_exist(s_)

        if r_type == 0:
            q_predict = self.q_table_access.loc[s, a]
            q_target = r + self.gamma * self.q_table_access.loc[s_, :].max() if s_ != 'terminal' else r
            self.q_table_access.loc[s, a] += self.lr * (q_target - q_predict)

        elif r_type == 1:
            q_predict = self.q_table_conflict.loc[s, a]
            q_target = r + self.gamma * self.q_table_conflict.loc[s_, :].max() if s_ != 'terminal' else r
            self.q_table_conflict.loc[s, a] += self.lr * (q_target - q_predict)

    def choose_action(self, observation):
        """
        不使用LSTM预测时的基础双Q选择函数。
        保留该函数是为了兼容原有调用方式。
        """
        self.check_state_exist(observation)

        q_values_access = self.q_table_access.loc[observation, :]
        q_values_conflict = self.q_table_conflict.loc[observation, :]
        weighted_q_values = 0.5 * q_values_access + 0.5 * q_values_conflict

        if np.random.uniform() < self.epsilon:
            weighted_q_values = weighted_q_values.reindex(np.random.permutation(weighted_q_values.index))
            action = weighted_q_values.idxmax()
        else:
            action = np.random.choice(self.actions)

        return action
