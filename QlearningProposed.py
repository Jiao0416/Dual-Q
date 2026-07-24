"""
文件名：QlearningProposed.py
方法名：Qlearning-Proposed

功能：单Q表 + LSTM 预测框架下的预测引导定向探索。
注意：epsilon 沿用当前项目语义，表示“利用概率”。
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models


pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


class QlearningProposed:
    def __init__(self, actions, num_channel, learning_rate=0.5, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.num_channel = num_channel
        self.channel_data = []

        self._build_lstm()
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    def _build_lstm(self):
        self.model = models.Sequential()
        self.model.add(layers.LSTM(units=50, activation='relu', input_shape=(10, self.num_channel)))
        self.model.add(layers.Dense(units=self.num_channel, activation='sigmoid'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Reshape((1, self.num_channel)))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    def channel_record(self, channel_state, access_act, action, su_index, num_su):
        channel_data_current = np.zeros(self.num_channel).astype(np.int32)

        for i in range(self.num_channel):
            channel_data_current[i] = channel_state[i]

        for i in range(num_su):
            if access_act[i] == 1 and i != su_index:
                channel_data_current[action[i]] = 1

        self.channel_data.append(channel_data_current)

    def channel_prediction(self, step):
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
        if state not in self.q_table.index:
            self.q_table = self.q_table._append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state)
            )

    def _q_values(self, observation):
        self.check_state_exist(observation)
        return self.q_table.loc[observation, :].astype(np.float64)

    def _argmax_action(self, q_values):
        q_values = q_values.reindex(np.random.permutation(q_values.index))
        return q_values.idxmax()

    def _lstm_directed_action(self, prediction):
        idle_probs = 1 - prediction[0]
        idle_probs = idle_probs + 1e-5
        prob_distribution = idle_probs / np.sum(idle_probs)
        return np.random.choice(self.actions, p=prob_distribution)

    def choose_action_lstm(self, observation, prediction):
        q_values = self._q_values(observation)

        if np.random.uniform() < self.epsilon:
            action = self._argmax_action(q_values)
        else:
            action = self._lstm_directed_action(prediction)

        return action

    def choose_action(self, observation):
        q_values = self._q_values(observation)

        if np.random.uniform() < self.epsilon:
            action = self._argmax_action(q_values)
        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]

        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)
