"""
文件名：DualQ_joint.py
方法名：DualQ-Joint

功能：
在双 Q 表 + LSTM 预测融合框架下，实现碰撞感知的联合增益探索。

核心思想：
1. 利用阶段：每个 SU 按 fused Q 最大值独立选择信道；
2. 探索阶段：先计算所有 SU-信道对的联合增益，再通过穷举选择总增益最大的无重复信道分配。
"""

from itertools import permutations

import numpy as np
import pandas as pd
from keras import layers, models
import tensorflow as tf


pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


class DualQJoint:
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

        self.q_table_access = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.q_table_conflict = pd.DataFrame(columns=self.actions, dtype=np.float64)

        self.num_channel = num_channel
        self.channel_data = []

        self._build_lstm()
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    def _build_lstm(self):
        """构建 LSTM 预测模型"""
        self.model = models.Sequential()
        self.model.add(layers.LSTM(units=50, activation='relu', input_shape=(10, self.num_channel)))
        self.model.add(layers.Dense(units=self.num_channel, activation='sigmoid'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Reshape((1, self.num_channel)))
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    def channel_record(self, channel_state, access_act, action, su_index, num_su):
        """记录 PU 占用状态和其他 SU 成功接入后的占用状态"""
        channel_data_current = np.zeros(self.num_channel).astype(np.int32)

        for i in range(self.num_channel):
            channel_data_current[i] = channel_state[i]

        for i in range(num_su):
            if access_act[i] == 1 and i != su_index:
                channel_data_current[action[i]] = 1

        self.channel_data.append(channel_data_current)

    def channel_prediction(self, step):
        """LSTM 在线训练与预测"""
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
        """检查状态是否存在，不存在则添加到两张 Q 表"""
        if state not in self.q_table_access.index:
            new_row = pd.Series([0] * len(self.actions), index=self.q_table_access.columns, name=state)
            self.q_table_access = self.q_table_access._append(new_row)

        if state not in self.q_table_conflict.index:
            new_row = pd.Series([0] * len(self.actions), index=self.q_table_conflict.columns, name=state)
            self.q_table_conflict = self.q_table_conflict._append(new_row)

    def get_fused_q_values(self, observation, prediction, num_channel):
        """
        计算 fused Q：
        Q_fused = Q_access * P_idle + Q_conflict * P_busy
        """
        self.check_state_exist(observation)

        q_values_access = self.q_table_access.loc[observation, :]
        q_values_conflict = self.q_table_conflict.loc[observation, :]
        busy_probs = prediction[0]
        idle_probs = 1 - busy_probs

        fused_q_values = (
            q_values_access * idle_probs +
            q_values_conflict * busy_probs
        )

        return fused_q_values.astype(np.float64)

    def choose_action_lstm(self, observation, prediction, num_channel):
        """
        兼容原有单 SU 调用方式：
        - epsilon 概率利用，选择 fused Q 最大信道；
        - 1-epsilon 概率按 idle probability 加权随机探索。

        新方法的联合探索应优先使用 joint_exploration_action()。
        """
        fused_q_values = self.get_fused_q_values(observation, prediction, num_channel)

        if np.random.uniform() < self.epsilon:
            fused_q_values = fused_q_values.reindex(np.random.permutation(fused_q_values.index))
            action = fused_q_values.idxmax()
        else:
            idle_probs = 1 - prediction[0]
            idle_probs = idle_probs + 1e-5
            prob_distribution = idle_probs / np.sum(idle_probs)
            action = np.random.choice(self.actions, p=prob_distribution)

        return action

    def choose_action(self, observation):
        """
        前期 LSTM 数据不足时使用的基础双 Q 选择函数。
        """
        self.check_state_exist(observation)

        q_values_access = self.q_table_access.loc[observation, :]
        q_values_conflict = self.q_table_conflict.loc[observation, :]
        fused_q_values = 0.5 * q_values_access + 0.5 * q_values_conflict

        if np.random.uniform() < self.epsilon:
            fused_q_values = fused_q_values.reindex(np.random.permutation(fused_q_values.index))
            action = fused_q_values.idxmax()
        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_, r_type):
        """双 Q 表更新：成功更新 Q_access，碰撞更新 Q_conflict"""
        self.check_state_exist(s_)

        if r_type == 0:
            q_predict = self.q_table_access.loc[s, a]
            q_target = r + self.gamma * self.q_table_access.loc[s_, :].max() if s_ != 'terminal' else r
            self.q_table_access.loc[s, a] += self.lr * (q_target - q_predict)

        elif r_type == 1:
            q_predict = self.q_table_conflict.loc[s, a]
            q_target = r + self.gamma * self.q_table_conflict.loc[s_, :].max() if s_ != 'terminal' else r
            self.q_table_conflict.loc[s, a] += self.lr * (q_target - q_predict)


def _positive_normalize(values):
    """
    将任意实数分数转换为概率分布。
    Q 值可能为负，因此先整体平移到非负区间，再归一化。
    """
    values = np.asarray(values, dtype=np.float64)
    values = values - np.min(values)
    values = values + 1e-5
    total = np.sum(values)

    if total <= 0 or np.any(np.isnan(values)):
        return np.ones_like(values) / len(values)

    return values / total


def build_joint_gain_matrix(agents, states, predictions, num_channel):
    """
    构造联合增益矩阵 G。

    对每个 SU k 和信道 i：
    base_score[k, i] = Q_fused(s_k, a_i) * P_idle(k, i)
    pi[k, i]         = normalize(base_score[k, :])
    P_no_col[k, i]   = prod_{l != k}(1 - pi[l, i])
    G[k, i]          = base_score[k, i] * P_no_col[k, i]
    """
    num_su = len(agents)
    fused_q_matrix = np.zeros((num_su, num_channel), dtype=np.float64)
    idle_prob_matrix = np.zeros((num_su, num_channel), dtype=np.float64)

    for k in range(num_su):
        fused_q_values = agents[k].get_fused_q_values(str(states[k]), predictions[k], num_channel)
        fused_q_matrix[k, :] = fused_q_values.to_numpy(dtype=np.float64)
        idle_prob_matrix[k, :] = 1 - predictions[k][0]

    base_score_matrix = fused_q_matrix * idle_prob_matrix

    pi_matrix = np.zeros_like(base_score_matrix)
    for k in range(num_su):
        pi_matrix[k, :] = _positive_normalize(base_score_matrix[k, :])

    no_collision_matrix = np.ones_like(base_score_matrix)
    for k in range(num_su):
        other_su_indices = [idx for idx in range(num_su) if idx != k]
        if other_su_indices:
            no_collision_matrix[k, :] = np.prod(1 - pi_matrix[other_su_indices, :], axis=0)

    gain_matrix = base_score_matrix * no_collision_matrix

    return gain_matrix, pi_matrix, no_collision_matrix


def joint_exploration_action(agents, states, predictions, num_channel):
    """
    碰撞感知联合增益探索。

    返回所有 SU 的联合动作 action：
    - 若 num_channel >= num_su，枚举无重复信道分配；
    - 若 num_channel < num_su，退化为允许重复的全组合枚举。
    """
    num_su = len(agents)
    gain_matrix, pi_matrix, no_collision_matrix = build_joint_gain_matrix(
        agents,
        states,
        predictions,
        num_channel
    )

    best_gain = -np.inf
    best_action = None

    if num_channel >= num_su:
        candidate_actions = permutations(range(num_channel), num_su)
    else:
        candidate_actions = np.ndindex(*(num_channel for _ in range(num_su)))

    for candidate_action in candidate_actions:
        total_gain = 0
        for k, channel in enumerate(candidate_action):
            total_gain += gain_matrix[k, channel]

        if total_gain > best_gain:
            best_gain = total_gain
            best_action = candidate_action

    return np.array(best_action, dtype=np.int32), gain_matrix, pi_matrix, no_collision_matrix
