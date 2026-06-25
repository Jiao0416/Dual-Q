"""
Q-Learning + LSTM + 加权定向探索
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models

# 设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


class QLSTM:
    def __init__(self, actions, num_channel, learning_rate=0.5, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # 动作列表
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        
        # === 单Q表设计 ===
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        
        self.num_channel = num_channel
        self.channel_data = []  # 信道历史记录
        
        # LSTM 模型与优化器
        self._build_lstm()
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)


    def _build_lstm(self):
        # 构建lstm预测模型
        # 输入数据 → [LSTM层] → [Dense层] → [Flatten] → [Reshape] → 输出
        self.model = models.Sequential()
        self.model.add(layers.LSTM(units=50, activation='relu', input_shape=(10, self.num_channel))) 
        self.model.add(layers.Dense(units=self.num_channel, activation='sigmoid')) 
        self.model.add(layers.Flatten())  
        self.model.add(layers.Reshape((1, self.num_channel)))  
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy']) 


    def channel_record(self, channel_state, access_act, action, su_index, num_su):
        # 记录信道状态
        channel_data_current = np.zeros(self.num_channel).astype(np.int32)
        for i in range(self.num_channel):
            channel_data_current[i] = channel_state[i]  
        for i in range(num_su):
            if access_act[i] == 1 and i != su_index:
                channel_data_current[action[i]] = 1
        self.channel_data.append(channel_data_current)  


    def channel_prediction(self, step):
        # lstm在线训练与预测
        if len(self.channel_data) < 11:
            return np.zeros((1, self.num_channel))

        x_train = np.expand_dims(self.channel_data[step - 11: step - 1], axis=0)  # (1,10,C)
        y_train = np.expand_dims(self.channel_data[step - 1: step], axis=0)       # (1,1,C)

        # === 手动训练：替代 model.fit() ===
        with tf.GradientTape() as tape:
            y_pred = self.model(x_train, training=True)  # 注意：training=True 启用 dropout/batchnorm
            loss = tf.keras.losses.mean_squared_error(y_train, y_pred)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # 预测下一步
        input_sequence = np.expand_dims(self.channel_data[step - 10: step], axis=0)
        prediction = np.squeeze(self.model(input_sequence, training=False), axis=0)
        return prediction
    

    def check_state_exist(self, state):  
        if state not in self.q_table.index:  
            self.q_table = self.q_table._append(
                pd.Series([0]*len(self.actions), index=self.q_table.columns,  name=state,)
            )


    def choose_action_lstm(self, observation, prediction):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # 选择Q值最大的动作
            q_values = self.q_table.loc[observation, :]
            weighted_q_values = q_values
            weighted_q_values = weighted_q_values.reindex(np.random.permutation(weighted_q_values.index))

            action = weighted_q_values.idxmax()

        else:
            # === 改进的探索策略：基于 LSTM 预测的加权定向探索 ===
            idle_probs = 1 - prediction[0]
            idle_probs = idle_probs + 1e-5
            prob_distribution = idle_probs / np.sum(idle_probs)
            action = np.random.choice(self.actions, p=prob_distribution)
            
        return action
    
    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            q_values = self.q_table.loc[observation, :]
            q_values = q_values.reindex(np.random.permutation(q_values.index))
            action = q_values.idxmax()
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