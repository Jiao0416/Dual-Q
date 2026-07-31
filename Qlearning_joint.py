"""
文件名：Qlearning_joint.py
方法名：Qlearning-Joint

功能：
在单 Q 表 + LSTM 预测框架下，实现碰撞感知的联合增益探索。

核心思想：
1. 利用阶段：每个 SU 按单 Q 表最大值独立选择信道；
2. 探索阶段：先计算所有 SU-信道对的联合增益，再通过穷举选择总增益最大的无重复信道分配。
"""

from itertools import permutations

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models


pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


class QlearningJoint:
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

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
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
        """检查状态是否存在，不存在则添加到 Q 表"""
        if state not in self.q_table.index:
            self.q_table = self.q_table._append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state)
            )

    def get_q_values(self, observation):
        """读取当前状态下的单 Q 表动作值"""
        self.check_state_exist(observation)
        return self.q_table.loc[observation, :].astype(np.float64)

    def choose_action_lstm(self, observation, prediction):
        """
        兼容原有单 SU 调用方式。
        新方法的联合探索应优先使用 joint_exploration_action()。
        """
        q_values = self.get_q_values(observation)

        if np.random.uniform() < self.epsilon:
            q_values = q_values.reindex(np.random.permutation(q_values.index))
            action = q_values.idxmax()
        else:
            idle_probs = 1 - prediction[0]
            idle_probs = idle_probs + 1e-5
            prob_distribution = idle_probs / np.sum(idle_probs)
            action = np.random.choice(self.actions, p=prob_distribution)

        return action

    def choose_action(self, observation):
        """前期 LSTM 数据不足时使用的基础 Q-learning 选择函数"""
        q_values = self.get_q_values(observation)

        if np.random.uniform() < self.epsilon:
            q_values = q_values.reindex(np.random.permutation(q_values.index))
            action = q_values.idxmax()
        else:
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        """单 Q 表更新"""
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]

        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)


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
    构造单 Q 框架下的联合增益矩阵 G。

    对每个 SU k 和信道 i：
    base_score[k, i] = Q(s_k, a_i) * P_idle(k, i)
    pi[k, i]         = normalize(base_score[k, :])
    P_no_col[k, i]   = prod_{l != k}(1 - pi[l, i])
    G[k, i]          = base_score[k, i] * P_no_col[k, i]
    """
    num_su = len(agents)
    q_matrix = np.zeros((num_su, num_channel), dtype=np.float64)
    idle_prob_matrix = np.zeros((num_su, num_channel), dtype=np.float64)

    for k in range(num_su):
        q_values = agents[k].get_q_values(str(states[k]))
        q_matrix[k, :] = q_values.to_numpy(dtype=np.float64)
        idle_prob_matrix[k, :] = 1 - predictions[k][0]

    base_score_matrix = q_matrix * idle_prob_matrix

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
