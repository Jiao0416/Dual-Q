"""
文件名：LSTM预_带屏蔽层版.py
功能：在原有双Q表+LSTM架构基础上，增加基于LSTM预测概率的信道屏蔽层
"""

import numpy as np
import pandas as pd
from keras import layers, models
import tensorflow as tf

# 设置显示参数
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

class DualQPlus:
    def __init__(self, actions, num_channel, learning_rate=0.5, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # 动作列表
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        
        # === 双Q表设计 ===
        self.q_table_access = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.q_table_conflict = pd.DataFrame(columns=self.actions, dtype=np.float64)
        
        self.num_channel = num_channel
        self.channel_data = []  # 信道历史记录
        
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
        # 注意：这里虽然compile了，但主要用GradientTape手动训练
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    def channel_record(self, channel_state, access_act, action, su_index, num_su):
        """记录信道状态"""
        channel_data_current = np.zeros(self.num_channel).astype(np.int32)
        # 记录PU状态
        for i in range(self.num_channel):
            channel_data_current[i] = channel_state[i]
        # 记录其他SU状态
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

        # 手动训练 (替代 model.fit)
        with tf.GradientTape() as tape:
            y_pred = self.model(x_train, training=True)
            loss = tf.keras.losses.mean_squared_error(y_train, y_pred)
            loss = tf.reduce_mean(loss)
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # 预测下一步
        input_sequence = np.expand_dims(self.channel_data[step - 10: step], axis=0)
        prediction = np.squeeze(self.model(input_sequence, training=False), axis=0)
        return prediction

    def check_state_exist(self, state):
        """检查状态是否存在，不存在则添加"""
        if state not in self.q_table_access.index:
            new_row = pd.Series([0]*len(self.actions), index=self.q_table_access.columns, name=state)
            self.q_table_access = self.q_table_access._append(new_row)
        if state not in self.q_table_conflict.index:
            new_row = pd.Series([0]*len(self.actions), index=self.q_table_conflict.columns, name=state)
            self.q_table_conflict = self.q_table_conflict._append(new_row)

    # ==========================================================================================
    # 🔑 核心修改点：choose_action_lstm (加入了信道屏蔽层)
    # ==========================================================================================
    def choose_action_lstm(self, observation, prediction, num_channel):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # 读取Q值
            q_values_access = self.q_table_access.loc[observation, :]
            q_values_conflict = self.q_table_conflict.loc[observation, :]

            # 加权计算 (算出综合Q值)
            ones = np.ones(num_channel)
            weighted_q_values = ( 
                q_values_access * (ones - prediction[0]) + 
                q_values_conflict * prediction[0] 
            )

            # # --- 🛡️ 硬屏蔽（信道屏蔽层Channel Masking Layer) 开始 ---
            # # 目的：如果LSTM预测信道占用概率过高，强制禁止选择该信道
            # RISK_THRESHOLD = 0.5  # 风险阈值：预测占用概率超过70%即视为禁区
            
            # # 遍历所有信道
            # for channel_idx in range(num_channel):
            #     # 如果LSTM预测该信道被占用的概率 > 阈值
            #     if prediction[0][channel_idx] > RISK_THRESHOLD: 
            #         # 强行将该信道的加权Q值设为负无穷
            #         # 这样在下一步取最大值时，绝对选不到它
            #         weighted_q_values.iloc[channel_idx] = -np.inf 
            # # ---  信道屏蔽层 结束 ---

            # # 3. 随机打乱与选择
            # # 处理极端情况：如果所有信道都被屏蔽了（全为-inf）
            # if weighted_q_values.max() == -np.inf:
            #     # 如果全被屏蔽，随机选一个（或者你可以选择不动作 'terminal'）
            #     action = np.random.choice(self.actions)
            # else:
            #     # 打乱索引，防止同分时总是选第一个
            #     weighted_q_values = weighted_q_values.reindex(np.random.permutation(weighted_q_values.index))
            #     action = weighted_q_values.idxmax()

            # 软屏蔽
            # SOFT_THRESHOLD = 0.7  # 固定温和阈值

            # for ch in range(num_channel):
            #     prob = prediction[0][ch]
            #     if prob > SOFT_THRESHOLD:
            #         # 风险越高，扣得越重，但不禁止选择
            #         weighted_q_values.iloc[ch] *= (1 - prob)
            
            # 选择最优动作
            weighted_q_values = weighted_q_values.reindex(np.random.permutation(weighted_q_values.index))
            action = weighted_q_values.idxmax()

            


        # else:
        #     # 探索模式：随机选择
        #     action = np.random.choice(self.actions)

        else:
            # === 改进的探索策略：基于预测的加权随机探索 ===
            # 计算空闲概率 (1 - 占用概率)
            idle_probs = 1 - prediction[0]

            # 防止除以0或概率全为0，加一个微小值
            idle_probs = idle_probs + 1e-5
            
            # 归一化，使其成为概率分布
            prob_distribution = idle_probs / np.sum(idle_probs)
            
            # 根据概率分布进行加权随机选择
            action = np.random.choice(self.actions, p=prob_distribution)
            
        return action

    # ==========================================================================================
    # 📉 学习函数 (保持不变)
    # ==========================================================================================
    def learn(self, s, a, r, s_, r_type):
        self.check_state_exist(s_)
        
        if r_type == 0:  # Success (Access)
            q_predict = self.q_table_access.loc[s, a]
            q_target = r + self.gamma * self.q_table_access.loc[s_, :].max() if s_ != 'terminal' else r
            self.q_table_access.loc[s, a] += self.lr * (q_target - q_predict)
            
        elif r_type == 1:  # Collision (Conflict)
            q_predict = self.q_table_conflict.loc[s, a]
            q_target = r + self.gamma * self.q_table_conflict.loc[s_, :].max() if s_ != 'terminal' else r
            self.q_table_conflict.loc[s, a] += self.lr * (q_target - q_predict)

    # ==========================================================================================
    # 📊 基础选择函数 (保持不变)
    # ==========================================================================================
    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            q_values_access = self.q_table_access.loc[observation, :]
            q_values_conflict = self.q_table_conflict.loc[observation, :]
            weighted_q_values = 0.5 * q_values_access + 0.5 * q_values_conflict
            weighted_q_values = weighted_q_values.reindex(np.random.permutation(weighted_q_values.index))
            action = weighted_q_values.idxmax()
        else:
            action = np.random.choice(self.actions)
        return action