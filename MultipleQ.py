"""
LSTM预测多类型概率分别与多类型Q值表相乘
"""

import numpy as np
import pandas as pd
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Reshape, Flatten
# from tensorflow.python.keras import models
from keras import layers, models


# 设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


class QLSTMTable:  

    # 初始化
    def __init__(self, actions, num_channel, learning_rate=0.5, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # 奖励折现因子
        self.epsilon = e_greedy  # 贪婪因子
        # 双Q表设计
        # 学习成功接入的经验，用户不与PU碰撞，创建表格，列名是动作，数据类型浮点数
        self.q_table_access = pd.DataFrame(columns=self.actions, dtype=np.float64)  
        # 学习碰撞惩罚的经验，用户与PU碰撞，创建表格，列名是动作，数据类型浮点数
        self.q_table_conflict = pd.DataFrame(columns=self.actions, dtype=np.float64)  
        self.num_channel = num_channel  # 可接入信道数量
        self.channel_data = []  # 信道实际接入记录
        self._build_lstm()
    

    
    # LSTM预测
    def _build_lstm(self):
        # 建立“空的”神经网络模型，按照我添加的顺序一层层处理数据
        # 输入数据 → [LSTM层] → [Dense层] → [Flatten] → [Reshape] → 输出
        self.model = models.Sequential()
        # 添加了：50个LSTM神经元，输入是10×num_channel的矩阵（每一个time_step中的特征值和维度）
        self.model.add(layers.LSTM(units=50, activation='relu', input_shape=(10, self.num_channel))) 
        # 添加了：输出层，num_channel个神经元，输出0-1之间概率值
        self.model.add(
            layers.Dense(units=self.num_channel, activation='sigmoid')) 
        # Flatten层，将输入扁平化
        self.model.add(layers.Flatten())  
        # Reshape层，调整形状
        self.model.add(layers.Reshape((1, self.num_channel)))  
        # 编译模型：配置模型的学习过程。指定优化器（optimizer）、损失函数（loss）、评估指标（metrics）、adam优化器（梯度下降法）、binary-cross entropy交叉熵适用于输出概率值情况、accuracy准确率
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy']) 

    """
    # 保存除当前用户外的其他用户的接入情况
    def channel_record(self, observation, action, su_index, num_su):
        channel_data_current = np.zeros(self.num_channel).astype(np.int32)
        for i in range(self.num_channel):
            channel_data_current[i] = observation[i]  # 记录当前pu占用信息
        for i in range(num_su):
            if i != su_index:
                channel_data_current[action[i]] = 1
        self.channel_data.append(channel_data_current)  # 将ACK接入情况和信道感知结果合并，然后添入到信道记录中，用于信道状态预测使用
    """

    
    # 记录当前时刻的完整信道占用状态，包括主用户（PU）和所有次用户（SU）的行为
    def channel_record(self, channel_state, access_act, action, su_index, num_su):
        # 假设所有信道都空闲
        channel_data_current = np.zeros(self.num_channel).astype(np.int32)
        # 先把主用户（PU）的占用情况写入 
        for i in range(self.num_channel):
            # 表示信道 i 被PU占用，记录为1
            channel_data_current[i] = channel_state[i]  
        for i in range(num_su):
            # access_act[i] == 1：表示第 i 个SU成功接入（收到了ACK）并且排除自己
            if access_act[i] == 1 and i != su_index:
                # 第i个SU选择的信道
                channel_data_current[action[i]] = 1
        self.channel_data.append(channel_data_current)  # 将ACK接入情况和信道感知结果合并，然后添入到信道记录中，用于信道状态预测使用



    # 在每次接入时，用最新的信道历史训练 LSTM 模型，并预测下一个时刻的信道状态（这个是否繁琐有待考虑，是否可以接入滑动窗口，训练数据量加大）
    def channel_prediction(self, step):
        # 因为 Keras LSTM 要求输入是 3D 张量：(batch_size, time_steps, features) (1, 10, num_channel)
        # x_train 是一个样本，包含过去 10 步的数据，用于预测第 step-1 时刻的状态
        x_train = np.expand_dims(self.channel_data[step - 11: step - 1], axis=0) 
        # 取第 step-1 时刻的信道状态
        y_train = np.expand_dims(self.channel_data[step - 1: step], axis=0)  
        # y_train = np.squeeze(y_train, axis=0)  # 将目标数据的形状从 (1, 1, 10) 调整为 (1, 10)

        # 训练模型
        # epochs=5：对这个样本重复训练 5 次（轻微过拟合，但有助于快速适应）
        # batch_size=1：每次只用一个样本更新权重 → 在线学习（Online Learning）
        # verbose=0：不打印训练日志，保持安静（0不显示，1显示进度条，2只显示损失）
        self.model.fit(x_train, y_train, epochs=5, batch_size=1, verbose=0)
        # print('channel_state', data[i + 1])
        # 用前十个时刻信道状态预测
        input_sequence = np.expand_dims(self.channel_data[step - 10: step], axis=0)  # 增加一个样本维度
        # 最终 prediction 是一个长度为 num_channel 的数组，表示每个信道被占用的概率（0~1之间）
        prediction = np.squeeze(self.model.predict(input_sequence, verbose=0), axis=0)
        # print('prediction', prediction)
        return prediction

   

    # 检查当前状态 state 是否已经在 Q 表中存在，如果不存在，就向 Q 表中添加一个新行，对应这个新状态
    def check_state_exist(self, state):  #
        if state not in self.q_table_access.index:  
            # 将这个新行添加到 Q 表末尾
            self.q_table_access = self.q_table_access._append(
                # 创建一个 Pandas Series（一维数组），表示新状态对应的 Q 值行
                pd.Series(
                    [0]*len(self.actions),  # 初始化为全0，长度为actions的长度
                    index=self.q_table_access.columns,  # 索引为列名
                    name=state,
                )
            )
        if state not in self.q_table_conflict.index:  
            self.q_table_conflict = self.q_table_conflict._append(
                pd.Series(
                    [0]*len(self.actions),  # 初始化为全0，长度为actions的长度
                    index=self.q_table_conflict.columns,  # 索引为列名
                    name=state,
                )
            )



    # 使用LSTM与QL相结合，双表双概率重新设计Q值表示，基于当前观测 observation 和 LSTM 预测结果 prediction，选择最佳动作
    def choose_action_lstm(self, observation, prediction, num_channel):

        # 确保当前状态 observation 已经在两个 Q 表中存在，如果是第一次遇到这个状态，会调用 check_state_exist 动态添加一行，Q 值初始化为 0
        self.check_state_exist(observation) 

        # 实现 ε-greedy 探索策略，生成一个 [0,1) 的随机数，如果小于 self.epsilon（如 0.9），
        # 则进入“利用模式”：选择当前看起来最优的动作；否则进入“探索模式”：随机选择动作，尝试新策略
        if np.random.uniform() < self.epsilon:  
            # 从两个 Q 表中分别读取当前状态下的所有动作 Q 值：
            q_values_access = self.q_table_access.loc[observation, :]  # 如果接入成功（没撞到 PU），能获得多少奖励？
            q_values_conflict = self.q_table_conflict.loc[observation, :]  # 如果接入失败（撞到 PU），会受到多少惩罚？

            # 使用LSTM预测的概率来加权Q值
            # 创建一个全 1 向量 [1, 1, ..., 1]，长度为 num_channel，用于后续计算“空闲概率”，1 - prediction[0] = 被空闲的概率
            ones = np.ones(num_channel)
            # print('ones:', ones)
            # print('access:', q_values_access)
            # 这里的计算还需要确认，空闲的表示到底是1还是0，是从信道占用角度看1是占用，用户的可用行方面来说1是空闲
            
            weighted_q_values = (
                    q_values_access * (ones - prediction[0]) +  # 空闲概率加权成功接入的Q值
                    q_values_conflict * prediction[0]  # 碰撞概率加权与碰撞的Q值
            )
            # 在多个动作 Q 值相同时，为了避免总是选择同一个动作（如索引最小的），引入随机性
            weighted_q_values = weighted_q_values.reindex(np.random.permutation(weighted_q_values.index)) 
            # 从打乱后的Q值中选择最大Q值动作
            action = weighted_q_values.idxmax()  
        else:
            action = np.random.choice(self.actions)
        return action
    
    """
    与 choose_action_lstm 的关键区别：
    choose_action	固定权重（0.5, 0.5）	 静态
    choose_action_lstm	LSTM 预测的占用概率	 动态
    """
    
    # 不使用 LSTM 预测的“基础版”决策函数，用于作对比，根据当前观测到的信道状态 observation，选择一个动作（即选择哪个信道接入）
    def choose_action(self, observation):
        self.check_state_exist(observation)  # 检查当前信道状态是否在Q值表中，若不在，添加新状态（本更新一般在第一次接入）
        # action selection 贪婪选择算法，一定概率不按照最高Q值选择
        if np.random.uniform() < self.epsilon:  # 均匀分布0-1，以贪婪因子概率选择Q值最高信道
            q_values_access = self.q_table_access.loc[observation, :]  # 成功接入的Q值
            q_values_conflict = self.q_table_conflict.loc[observation, :]  # 碰撞的Q值
            # choose best action
            weighted_q_values = (
                    0.5 * q_values_access +  # 空闲概率加权成功接入的Q值
                    0.5 * q_values_conflict  # 碰撞概率加权与碰撞的Q值
            )
            # 对选定的行重新索引 np.random.permutation对索引进行重新排序，相同Q值时随机选择其中一个
            weighted_q_values = weighted_q_values.reindex(np.random.permutation(weighted_q_values.index))  # 一些动作有相同的值
            action = weighted_q_values.idxmax()  # 从打乱后的Q值中选择最大Q值动作
        else:
            # choose random action 否则随机选择信道 探索概率
            action = np.random.choice(self.actions)
        return action


    
    # 根据一次交互的经验进行学习（即“更新 Q-learningn中的Q值”）
    def learn(self, s, a, r, s_, r_type):
        # 确保下一个状态 s_ 已经在两个 Q 表中存在， 若不在，添加新状态（本更新一般在除第一次外，每次信道更新后在计算Q值前会添加）
        self.check_state_exist(s_)  
        # q_predict = self.q_table.loc[s, a]   # 当前对应的Q值

        # SU 成功接入且没有干扰主用户（PU），更新 q_table_access
        if r_type == 0:  
            # 获取当前状态 s 下动作 a 的预测 Q 值（即旧值）
            q_predict_access = self.q_table_access.loc[s, a] 
            if s_ != 'terminal':  # 判断迭代是否结束，未结束，计算学习结果
                q_target_access = r + self.gamma * self.q_table_access.loc[s_, :].max()  
            else:  # 否则，计算结果不改变
                q_target_access = r
            self.q_table_access.loc[s, a] += self.lr * (q_target_access - q_predict_access)

        # 表示 SU 接入时与 PU 发生了碰撞（干扰了主用户），更新 q_table_conflict
        elif r_type == 1:  # 碰撞且接入
            q_predict_conflict = self.q_table_conflict.loc[s, a]  # 与PU碰撞的负惩罚
            if s_ != 'terminal':  # 判断迭代是否结束，未结束，计算学习结果
                q_target_conflict = r + self.gamma * self.q_table_conflict.loc[s_, :].max()  # 与PU碰撞的Q值
            else:  # 否则，计算结果不改变
                q_target_conflict = r
            self.q_table_conflict.loc[s, a] += self.lr * (q_target_conflict - q_predict_conflict)
        # self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update 更新Q值 Q+lr*计算结果
        # 更新三张Q值表


