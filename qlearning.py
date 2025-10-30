"""
Q-Learning算法设计
"""

import numpy as np
import pandas as pd

# 设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


class QLearningTable:  # Q值表
    def __init__(self, actions, learning_rate=0.5, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate  # 学习率
        self.gamma = reward_decay  # 奖励折现因子
        self.epsilon = e_greedy  # 贪婪因子
        # 创建一个空的 Q 表：
        # 每一行代表一个状态（如 '1010' 表示信道占用情况）
        # 每一列代表一个动作（信道编号）
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)  # 创建表格，列名是动作，数据类型浮点数



    def check_state_exist(self, state):  
        if state not in self.q_table.index:  
            # append new state to q table
            self.q_table = self.q_table._append(
                pd.Series(
                    [0]*len(self.actions),  # 初始化为全0，长度为actions的长度
                    index=self.q_table.columns,  # 索引为列名
                    name=state,
                )
            )



    def choose_action(self, observation):
        self.check_state_exist(observation)  # 检查当前信道状态是否在Q值表中，若不在，添加新状态（本更新一般在第一次接入）
        # action selection 贪婪选择算法，一定概率不按照最高Q值选择
        if np.random.uniform() < self.epsilon:  # 均匀分布0-1，以贪婪因子概率选择Q值最高信道
            # choose best action
            state_action = self.q_table.loc[observation, :]  # 获取当前状态和动作的Q值
            # 对选定的行重新索引 np.random.permutation对索引进行重新排序，相同Q值时随机选择其中一个
            state_action = state_action.reindex(np.random.permutation(state_action.index))  # 一些动作有相同的值
            action = state_action.idxmax()  # 从打乱后的Q值中选择最大Q值动作
        else:
            # choose random action 否则随机选择信道 探索概率
            action = np.random.choice(self.actions)
        return action



    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)  # 检查下一时刻信道状态是否在Q值表中,若不在，添加新状态（本更新一般在除第一次外，每次信道更新后在计算Q值前会添加）
        q_predict = self.q_table.loc[s, a]   # 当前对应的Q值
        if s_ != 'terminal':  # 判断迭代是否结束，未结束，计算学习结果
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:  # 否则，计算结果不改变
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update 更新Q值 Q+lr*计算结果