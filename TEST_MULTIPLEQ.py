"""
overlay模式下双Q网络的仿真实验，与Qlearning算法进行对比
"""
import numpy as np
np.random.seed(7) # 设置随机种子，确保实验结果可重复
import os
from Env import Environment # 导入环境类
# from qlstm import QLearningWithLSTMTable
import matplotlib.pyplot as plt
import copy
import pandas as pd
from qlearning import QLearningTable # 传统Q学习算法
from MultipleQ import QLSTMTable     # 论文提出的双Q+LSTM算法

# from DQN import MLP1

# from MultipleDQN import MMLP1


if __name__ == "__main__":

    # ========== 环境参数设置 ==========
    num_channel = 8
    num_pu = 4
    num_su = 6

    # 初始化通信环境
    env = Environment(num_channel, num_su, num_pu)  # 创建环境对象
    # print('channel_state', env.channel_state)
    env_copy = copy.deepcopy(env)  # 深拷贝，用于公平对比实验


    # ========== 训练参数设置 ==========
    batch_size = 100
    episodes = 5
    replace_target_iter = 1
    total_episode = batch_size * replace_target_iter * episodes
    epsilon_update_period = batch_size * replace_target_iter * 10
    e_greedy = [0.3, 0.85]  # [0.3, 0.9, 1]
    # e_greedy = [0.85, 0.3]
    learning_rate = 0.1
    # step 是最小单位的动作，batch_size 是你用来观察趋势的时间窗口（多少个step总结一次）


    # ========== 实验标志设置 ==========
    # flag_QLearning = False
    # flag_MultipleQLearning = False
    # flag_random = False

    flag_QLearning = True
    flag_MultipleQLearning = True
    flag_random = True



    # ========== 传统Q学习算法实验 ==========
    if flag_QLearning:
        # 为了公平对比，使用相同的环境初始状态
        env = copy.deepcopy(env_copy)

        # 初始化每个SU的Q表
        QL_list = []   # 是创建一个空列表，用来存放所有 SU 的 Q 学习器实例。
        epsilon_index = np.zeros(num_su, dtype=int)  # 每个SU的贪婪因子，每个SU的探索率索引

        # 为每个SU创建独立的Q学习器（每个SU有自己的Q表，每个 SU 都有自己独立的学习策略，可以有不同的行为模式）
        for k in range(num_su):
            QL_tmp = QLearningTable(
                actions=list(range(env.n_actions)),  # 动作空间：选择哪个信道
                learning_rate=learning_rate,         # 学习率
                reward_decay=0.9,                    # 折扣因子
                e_greedy=e_greedy[0])                # 初始探索率
            QL_list.append(QL_tmp)                   # 把每个新创建的 Q 学习器加入列表
            # QL_list[k]：获取第 k 个 SU 的 Q 学习器


        # SU感知环境，获得信道状态观测值
        observation = env.sense()  # 返回num_su * num_channel 的矩阵序列

        # 初始化状态记录
        # 创建包含num_su个空列表的列表
        state = [[] for i in range(num_su)]   # 当前时刻每个SU看到的状态
        # state = [[], [], [], [], [], [], []]  # 6 个空的小列表
        state_ = [[] for i in range(num_su)]  # 下一时刻每个SU看到的状态

        # print('sense_return')
        """
        # 输出每个su感知到的信道状态
        for state_list in state:
            print(state_list)
        """
        # ========== 性能记录初始化 ==========
        reward_sum = np.zeros(num_su)        # 累计奖励
        overall_reward = []                  # 总体奖励历史
        success_history = []                 # 成功接入历史
        fail_PU_history = []                 # PU碰撞历史
        fail_collision_history = []          # SU碰撞历史
        QoS_history = []                     # 网络容量历史
        success_sum = 0                      # 成功接入计数
        fail_PU_sum = 0                      # PU碰撞计数
        fail_collision_sum = 0               # SU碰撞计数
        QoS_sum = 0                          # 网络容量累计


        # 每个SU的动作选择（0,1,...,num_channel-1）
        action = np.zeros(num_su).astype(np.int32)  # 数组类型转换为32位整数 (1 * num_su)


        # ========== Q实验主训练循环 ==========
        for step in range(total_episode):
            # print('channel_state', env.channel_state)
            # channel_data.append(observation[0, :])  信道记录
            # 给状态序列赋值，初始化感知情况
            # 1. 获取当前状态
            for k in range(num_su):
                state[k] = observation[k, :]  # 将全局感知拆分为每个 SU 的局部状态，
                                              # state[k] = 第 k 个 SU 的当前状态
             
            np.set_printoptions(threshold=np.inf)  # 打印选项设置为完全显示数组内容
            # print('用户1：Q值', QL_list[0].q_table)
            # print('用户2：Q值', QL_list[1].q_table)
            # print('用户3：Q值', QL_list[2].q_table)
            # print('用户4：Q值', QL_list[3].q_table)
            # SU choose action based on observation
            # print('state', state[0])

            # 2. SU基于当前状态选择动作
            for k in range(num_su):
                action[k] = QL_list[k].choose_action(str(state[k]))  # 用户k面对当前状态采取的动作选择
            # 第 k 个次用户（SU）根据自己当前看到的信道状态 state[k]，通过其专属的 Q 学习器 QL_list[k] 来决定下一步要做什么（比如选择哪个信道进行接入），其中 str(state[k]) 是为了把状态转成字符串，作为 Q 表的“行索引”

            # print('Q', QL_list[0].q_table)
            # print('action', action)

            # 3. 执行动作并获得环境反馈
            reward, QoS, access_act, reward_type = env.access(action)  
            # print('action_act', action_act)
            # reward: 每个SU的即时奖励
            # QoS: 网络总容量
            # access_act: 实际成功接入的SU标记
            # reward_type: 奖励类型（0=成功，1=碰撞）


            # 4. 记录性能指标
            reward_sum = reward_sum + reward
            success_sum = success_sum + env.success
            fail_PU_sum = fail_PU_sum + env.fail_PU
            fail_collision_sum = fail_collision_sum + env.fail_collision
            QoS_sum = QoS_sum + QoS


            # 5. 环境更新（马尔可夫信道状态变化）
            env.render()  
            # print('channel_state', env.channel_state)

            # 6. SU感知新的环境状态，获取新的观测值
            observation_ = env.sense() 

            # 7. 获取下一状态 (s, a, r, s_)
            for k in range(num_su):
                # state[k] = observation[k, :]  # 将每个 SU 的新观测值赋值给它的“下一状态”变量 state_
                state_[k] = observation_[k, :]

            # 8. Q学习更新
            for k in range(num_su):
                QL_list[k].learn(str(state[k]), action[k], reward[k], str(state_[k]))
            # 告诉第 k 个 SU 智能体：刚才的状态是 state[k]，选择了动作 action[k]，得到了奖励 reward[k]，然后环境变成了新状态 state_[k]，请根据这个经验，更新 Q 表！
            # def learn(self, s, a, r, s_):
            # Q(s,a) ← Q(s,a) + α [ r + γ * max_a' Q(s',a') - Q(s,a) ]
            
            
            # 9. 定期记录和输出
            if (step + 1) % batch_size == 0:
                # 计算批次平均值并记录
                overall_reward.append(np.sum(reward_sum) / batch_size / num_su)
                # reward_sum，在这 batch_size 步中，每个 SU 累计获得的奖励总和（是一个数组）；np.sum(reward_sum)	所有 SU 的总奖励之和；最终得到：平均每步、每个 SU 获得的奖励
                success_history.append(success_sum / num_su)  # / batch_size
                fail_PU_history.append(fail_PU_sum / num_su)  # / batch_size
                fail_collision_history.append(fail_collision_sum / num_su)  # / batch_size
                QoS_history.append(QoS_sum / batch_size)  # 网络容量

                # After one batch, 重置计数器
                reward_sum = np.zeros(num_su)
                success_sum = 0
                fail_PU_sum = 0
                fail_collision_sum = 0
                QoS_sum = 0
                # [step 0] → [step 1] → ... → [step 99]  
                #        ↑                        ↓
                #     开始累加               满100步？✅是！
                #                  ↓
                #          记录这100步的平均表现
                #                  ↓
                #           清零所有 sum 变量
                #                  ↓
                # [step 100] → [step 101] → ... → [step 199]
                #         ↑                        ↓
                #       重新累加             再次满100步？✅是！
                #                  ↓
                #           再次记录 + 再次清零
                # 若每一个step都记录，则数据太多、噪声大、绘图杂乱


            # 10. 更新探索率（随时间减少探索，增加利用）
            if ((step + 1) % epsilon_update_period == 0):
                # 每个 SU 都有自己的 Q-learning 策略和 ε 值，要对每一个 SU 单独更新它的探索率。
                for k in range(num_su):
                    epsilon_index[k] = min(len(e_greedy) - 1, epsilon_index[k] + 1)
                    # 把当前索引 +1 → 表示“升级”到下一个更低的 ε，但不能超过最大索引（防止越界）
                    QL_list[k].epsilon = e_greedy[epsilon_index[k]]
                    # 真正更新 ε 值，把新的 ε 赋值给第 k 个 Q-learning 智能体
                print('epsilon update to %.1f' % (QL_list[k].epsilon))
                # 打印当前最新的 ε 值，方便你在控制台看到
            # 每过 epsilon_update_period 步，就降低一次智能体的随机性（ε 值），让它从‘大胆尝试’逐渐变成‘只做最优选择’
            # 目的是：先多探索 → 后多利用

            # 11. 进度输出
            if (step + 1) % batch_size == 0:
                # print('Training time = %d;  success = %d;  fail_PU = %d;  fail_collision = %d' % ((step + 1), success_history[-1]*num_su, fail_PU_history[-1]*num_su, fail_collision_history[-1]*num_su))
                print('Training time = %d;  success = %d;  fail_PU = %d;  fail_collision = %d; overall_reward = %.4f' %
                      ((step + 1), success_history[-1], fail_PU_history[-1], fail_collision_history[-1],
                       overall_reward[-1]))
                # [-1] 表示取最后一个元素（即最新一个批次的数据）
                # print('overall_reward = %.4f' % overall_reward[-1])

            # 12. 状态流转：当前观测变为下一时刻的观测
            observation = observation_

        # channel_data = np.array(channel_data)  信道记录
        # print('channel_data', channel_data)
        # print('length', len(channel_data))
        

        
        # ========== 结果保存 ==========

        # 使用 os.path.join 自动适配平台（Mac用/, Windows用\）
        file_folder = os.path.join('result', 'overlay', f'channel_{num_channel}_su_{num_su}_punish_-2_-10_Q')

        # 创建目录
        os.makedirs(file_folder, exist_ok=True)

        # 保存文件时也用 os.path.join
        np.save(os.path.join(file_folder, 'success_history'), success_history)
        np.save(os.path.join(file_folder, 'fail_PU_history'), fail_PU_history)
        np.save(os.path.join(file_folder, 'fail_collision_history'), fail_collision_history)
        np.save(os.path.join(file_folder, 'overall_reward'), overall_reward)
        np.save(os.path.join(file_folder, 'QoS_history'), QoS_history)

        # 保存为 Excel
        df = pd.DataFrame(success_history)
        df.to_excel(os.path.join(file_folder, 'success_history.xlsx'), index=False)

        df = pd.DataFrame(fail_PU_history)
        df.to_excel(os.path.join(file_folder, 'fail_PU_history.xlsx'), index=False)

        df = pd.DataFrame(fail_collision_history)
        df.to_excel(os.path.join(file_folder, 'fail_collision_history.xlsx'), index=False)

        df = pd.DataFrame(overall_reward)
        df.to_excel(os.path.join(file_folder, 'overall_reward.xlsx'), index=False)

        df = pd.DataFrame(QoS_history)
        df.to_excel(os.path.join(file_folder, 'QoS_history.xlsx'), index=False)



    


    # ========== 论文提出的双Q+LSTM算法实验 ==========
    if flag_MultipleQLearning:
        # 使用相同的环境初始状态
        env = copy.deepcopy(env_copy)

        # 初始化每个SU的双Q+LSTM学习器
        QL_list = []  # 每个SU的Q值列表综合
        epsilon_index = np.zeros(num_su, dtype=int)  # 初始化每个SU的贪婪因子
        for k in range(num_su):
            QL_tmp = QLSTMTable(
                actions=list(range(env.n_actions)),
                num_channel=num_channel, # 需要信道数量信息来构建LSTM
                learning_rate=learning_rate, 
                reward_decay=0.9,
                e_greedy=e_greedy[0])  # 此处已设置channel_data、model_lstm
            QL_list.append(QL_tmp)

        # 初始感知，创建包含num_su个空列表的列表
        observation = env.sense()  # 返回num_su * num_channel 的序列
        state = [[] for i in range(num_su)]  # 每个SU看到的当前信道状态
        state_ = [[] for i in range(num_su)]  # 下一时刻信道状态

        # print('sense_return')
        """
        # 输出每个su感知到的信道状态
        for state_list in state:
            print(state_list)
        """
        # 性能记录初始化（同传统Q学习）
        reward_sum = np.zeros(num_su)
        overall_reward = []
        success_history = []
        fail_PU_history = []
        fail_collision_history = []
        QoS_history = []
        success_sum = 0
        fail_PU_sum = 0
        fail_collision_sum = 0
        QoS_sum = 0

        # 每个SU采取的动作（0,1,...,num_channel）
        action = np.zeros(num_su).astype(np.int32)  # 数组类型转换为32位整数 (1 * num_su)
        # batch * episode 次迭代


        # ========== 主训练循环（与传统的区别） ==========
        for step in range(total_episode):
            # 每10个step打印一次
            if step % 10 == 0:
                print(f"[Step {step}] SU0预测动作: {action[0]}")
            # 1.获取当前状态
            for k in range(num_su):
                state[k] = observation[k, :]  # 用户k看到的信道状态

            np.set_printoptions(threshold=np.inf)  # 打印选项设置为完全显示数组内容
            # print('用户1：Q值', QL_list[0].q_table)
            # print('用户2：Q值', QL_list[1].q_table)
            # print('用户3：Q值', QL_list[2].q_table)
            # print('用户4：Q值', QL_list[3].q_table)
            # SU choose action based on observation
            # print('state', state[0])

            # 2.动作选择（关键区别！）
            for k in range(num_su):
                if step > 10:  # 前10步用传统方法，积累足够数据后再用LSTM
                    # print(QL_list[k].channel_data)
                    # 使用LSTM预测加权双Q值的智能决策
                    prediction = QL_list[k].channel_prediction(step)
                    # 调用 LSTM 模型，用过去 10 步的完整信道占用记录来预测未来
                    # 假设：prediction = [0.95, 0.20, 0.80]，查双 Q 表（经验），动态加权，计算综合 Q 值，weighted_Q = Q_access * (1 - p) + Q_conflict * p，选择动作
                    action[k] = QL_list[k].choose_action_lstm(
                        str(state[k]), 
                        prediction,
                        num_channel)
                    # 需要接入 prediction 是因为是动态变化的，每信道独立动态加权，传统的0.5/0.5是静态的，num_channel是因为ones = np.ones(num_channel)，目的是生成一个和 prediction 长度相同的全1向量，用于计算 1 - prediction（即空闲概率），也为了代码更健壮
                else: # 前10步使用传统Q学习
                    action[k] = QL_list[k].choose_action(str(state[k]))  # ← 前10步用传统方法

            # print('Q', QL_list[0].q_table)
            # print('action', action)

            # 3.执行动作并获得环境反馈
            reward, QoS, access_act, reward_type = env.access(action)  # 执行接入动作，获得奖励列表（1 * num_su）
            # print('access_act', access_act)

            # 4.记录信道数据（LSTM 训练的关键！）
            # 因为 LSTM 预测需要高质量的信道占用历史数据，而单靠本地 observation 无法获知“其他 SU 是否成功占用了信道”。通过 access_act 和 action，每个 SU 能重建接近真实的全局占用状态，从而训练出准确的预测模型，最终实现更智能的频谱接入决策
            for su_index in range(num_su):
                # su_index：当前正在处理的第几个 SU，第1个 SU → su_index = 0，
                # 让当前这个 SU 记录自己的信道历史
                QL_list[su_index].channel_record(
                    observation[su_index],    # 记录每个 SU 自己感知到的 PU 占用情况
                    access_act,               # 哪些 SU 成功接入了（收到 ACK）
                    action,                   # 所有 SU 这一步选择了哪个信道
                    su_index,                 # 当前 SU 编号
                    num_su)                   # 总 SU 数
            # QL_list[1].channel_record(
            # [1, 0, 0, 0],   # 第2个SU看到：PU 占了信道0
            # [1, 0, 1],      # 第1个SU 和 第3个SU 成功，第2个SU 失败
            # [1, 2, 1],      # 第1个SU选1，第2个SU选2，第3个选1
            # 1,              # 第1个SU
            # 3               # 总共3个SU
            # 最终[1, 1, 0, 0]
            # QL_list[su_index] 不只是一个 Q 表，而是一个完整的智能体对象；channel_record 是它的一个方法，专门用来记录信道占用历史，为后续的 LSTM 预测提供数据支持


            # 5.累加性能指标（同传统 Q）
            reward_sum = reward_sum + reward
            success_sum = success_sum + env.success
            fail_PU_sum = fail_PU_sum + env.fail_PU
            fail_collision_sum = fail_collision_sum + env.fail_collision
            QoS_sum = QoS_sum + QoS

            # 6.环境更新
            env.render()  # 马尔可夫信道更新
            # print('channel_state', env.channel_state)
            observation_ = env.sense()  # 感知更新后的信道

            # 7.获取下一状态 (s, a, r, s_)
            for k in range(num_su):
                # state[k] = observation[k, :]  # 重新赋值，第一次循环之后的所有循环都需要改变此处状态
                state_[k] = observation_[k, :]
            # print('---------------------------------------')

            # 8.双 Q 学习更新（关键区别2）（根据奖励类型更新不同的Q表）
            for k in range(num_su):
                QL_list[k].learn(str(state[k]), action[k], reward[k], str(state_[k]), reward_type[k])
            # 正是因为有两个 Q 表（Q_access 和 Q_conflict），才需要传入 reward_type[k] 来指示：当前这个奖励应该用来更新哪一张 Q 表

            # 9.定期记录性能（每 batch_size 步）同传统）
            if (step + 1) % batch_size == 0:
                # Record reward, the number of success / interference / collision
                overall_reward.append(np.sum(reward_sum) / batch_size / num_su)
                success_history.append(success_sum / num_su)  # / batch_size
                fail_PU_history.append(fail_PU_sum / num_su)  # / batch_size
                fail_collision_history.append(fail_collision_sum / num_su)  # / batch_size
                QoS_history.append(QoS_sum / batch_size)  # 网络容量

                # After one batch, refresh the record
                reward_sum = np.zeros(num_su)
                success_sum = 0
                fail_PU_sum = 0
                fail_collision_sum = 0
                QoS_sum = 0

            # 10.更新探索率 ε（同传统）
            if ((step + 1) % epsilon_update_period == 0):
                for k in range(num_su):
                    epsilon_index[k] = min(len(e_greedy) - 1, epsilon_index[k] + 1)
                    QL_list[k].epsilon = e_greedy[epsilon_index[k]]
                print('epsilon update to %.1f' % (QL_list[k].epsilon))
            
            # 11.打印进度 + 状态流转
            # Print the record after replace DQN_target
            if (step + 1) % batch_size == 0:
                print('step', step)
                # print('Training time = %d;  success = %d;  fail_PU = %d;  fail_collision = %d' % ((step + 1), success_history[-1]*num_su, fail_PU_history[-1]*num_su, fail_collision_history[-1]*num_su))
                print('Training time = %d;  success = %d;  fail_PU = %d;  fail_collision = %d;  overall_reward = %.4f' %
                      ((step + 1), success_history[-1], fail_PU_history[-1], fail_collision_history[-1],
                       QoS_history[-1]))
                # print('overall_reward = %.4f' % overall_reward[-1])

            # 状态流转
            observation = observation_
        """
        channel_data = np.array(channel_data)
        print('channel_data', channel_data)
        print('length', len(channel_data))
        """


        # ========== 结果保存 ==========

        # 使用 os.path.join 自动适配平台（Mac用/, Windows用\）
        file_folder = os.path.join('result', 'overlay', f'channel_{num_channel}_su_{num_su}_punish_-2_-10_MULTIQ')

        # 创建目录
        os.makedirs(file_folder, exist_ok=True)

        # 保存文件时也用 os.path.join
        np.save(os.path.join(file_folder, 'success_history'), success_history)
        np.save(os.path.join(file_folder, 'fail_PU_history'), fail_PU_history)
        np.save(os.path.join(file_folder, 'fail_collision_history'), fail_collision_history)
        np.save(os.path.join(file_folder, 'overall_reward'), overall_reward)
        np.save(os.path.join(file_folder, 'QoS_history'), QoS_history)

        # 保存为 Excel
        df = pd.DataFrame(success_history)
        df.to_excel(os.path.join(file_folder, 'success_history.xlsx'), index=False)

        df = pd.DataFrame(fail_PU_history)
        df.to_excel(os.path.join(file_folder, 'fail_PU_history.xlsx'), index=False)

        df = pd.DataFrame(fail_collision_history)
        df.to_excel(os.path.join(file_folder, 'fail_collision_history.xlsx'), index=False)

        df = pd.DataFrame(overall_reward)
        df.to_excel(os.path.join(file_folder, 'overall_reward.xlsx'), index=False)

        df = pd.DataFrame(QoS_history)
        df.to_excel(os.path.join(file_folder, 'QoS_history.xlsx'), index=False)

        
    




    # ========== 随机策略基准实验 ==========
    # 设置探索率为0，即完全随机选择
    # 代码结构与传统Q学习相同，只是探索率固定为0
    # 用于作为性能比较的基准线
    if flag_random:
        # For fair comparison, initialize the DSA environment with the same properties
        env = copy.deepcopy(env_copy)

        # Initialize the Q-table for each SU
        QL_list = []  # 每个SU的Q值列表综合
        epsilon_index = np.zeros(num_su, dtype=int)  # 每个SU的贪婪因子
        for k in range(num_su):
            QL_tmp = QLearningTable(
                actions=list(range(env.n_actions)), 
                learning_rate=learning_rate, 
                reward_decay=0.9,
                e_greedy=0) # e_greedy=0 而不是 e_greedy=e_greedy[0] e_greedy=0 表示：永远不利用 Q 表，100% 随机选动作！虽然叫 QLearningTable，但Q 表在这里完全没用，只是复用其 choose_action 接口来实现随机选择。所以这本质上是一个 “披着 Q-learning 外衣的纯随机策略”。
            QL_list.append(QL_tmp)

        observation = env.sense()  # 返回num_su * num_channel 的序列

        # 创建包含num_su个空列表的列表
        state = [[] for i in range(num_su)]  # 每个SU看到的当前信道状态
        state_ = [[] for i in range(num_su)]  # 下一时刻信道状态

        # print('sense_return')
        """
        # 输出每个su感知到的信道状态
        for state_list in state:
            print(state_list)
        """
        # Initialize some record values
        reward_sum = np.zeros(num_su)
        overall_reward = []
        success_history = []
        fail_PU_history = []
        fail_collision_history = []
        QoS_history = []
        success_sum = 0
        fail_PU_sum = 0
        fail_collision_sum = 0
        QoS_sum = 0

        # 每个SU采取的动作（0,1,...,num_channel-1）
        action = np.zeros(num_su).astype(np.int32)  # 数组类型转换为32位整数 (1 * num_su)
      

        for step in range(total_episode):
            # print('channel_state', env.channel_state)
            # channel_data.append(observation[0, :])  信道记录
            # 给状态序列赋值，初始化感知情况
            for k in range(num_su):
                state[k] = observation[k, :]  # 用户k看到的信道状态

            np.set_printoptions(threshold=np.inf)  # 打印选项设置为完全显示数组内容
            # print('用户1：Q值', QL_list[0].q_table)
            # print('用户2：Q值', QL_list[1].q_table)
            # print('用户3：Q值', QL_list[2].q_table)
            # print('用户4：Q值', QL_list[3].q_table)
            # SU choose action based on observation
            # print('state', state[0])

            for k in range(num_su):
                action[k] = QL_list[k].choose_action(str(state[k]))  # 用户k面对当前状态采取的动作选择

            # print('Q', QL_list[0].q_table)
            # print('action', action)

            # SU take action and get the reward
            reward, QoS, access_act, reward_type = env.access(action)  # 执行接入动作，获得奖励列表（1 * num_su）
            # print('action_act', action_act)

            # Record reward
            reward_sum = reward_sum + reward
            success_sum = success_sum + env.success
            fail_PU_sum = fail_PU_sum + env.fail_PU
            fail_collision_sum = fail_collision_sum + env.fail_collision
            QoS_sum = QoS_sum + QoS

            env.render()  # 马尔可夫信道更新
            # print('channel_state', env.channel_state)

            observation_ = env.sense()  # 感知更新后的信道

            # Store one episode (s, a, r, s_)
            for k in range(num_su):
                # state[k] = observation[k, :]  # 重新赋值，第一次循环之后的所有循环都需要改变此处状态
                state_[k] = observation_[k, :]

            # Each SU learns their QL model
            # SU的Q值表更新
            for k in range(num_su):
                QL_list[k].learn(str(state[k]), action[k], reward[k], str(state_[k]))
                # 即使调用了 learn，但由于动作始终随机，Q 表无法学到有效策略。本质上，Q 表是“摆设”，系统行为完全由随机动作驱动。

            if (step + 1) % batch_size == 0:
                # Record reward, the number of success / interference / collision
                overall_reward.append(np.sum(reward_sum) / batch_size / num_su)
                success_history.append(success_sum / num_su)  # / batch_size
                fail_PU_history.append(fail_PU_sum / num_su)  # / batch_size
                fail_collision_history.append(fail_collision_sum / num_su)  # / batch_size
                QoS_history.append(QoS_sum / batch_size)  # 网络容量

                # After one batch, refresh the record
                reward_sum = np.zeros(num_su)
                success_sum = 0
                fail_PU_sum = 0
                fail_collision_sum = 0
                QoS_sum = 0

            """
            # Update epsilon
            if ((step + 1) % epsilon_update_period == 0):
                for k in range(num_su):
                    epsilon_index[k] = min(len(e_greedy) - 1, epsilon_index[k] + 1)
                    QL_list[k].epsilon = e_greedy[epsilon_index[k]]
                print('epsilon update to %.1f' % (QL_list[k].epsilon))
            """

            # Print the record after replace DQN_target
            if (step + 1) % batch_size == 0:
                # print('Training time = %d;  success = %d;  fail_PU = %d;  fail_collision = %d' % ((step + 1), success_history[-1]*num_su, fail_PU_history[-1]*num_su, fail_collision_history[-1]*num_su))
                print('Training time = %d;  success = %d;  fail_PU = %d;  fail_collision = %d; overall_reward = %.4f' %
                      ((step + 1), success_history[-1], fail_PU_history[-1], fail_collision_history[-1],
                       overall_reward[-1]))
                # print('overall_reward = %.4f' % overall_reward[-1])

            # swap observation 信道状态流转
            observation = observation_

        # channel_data = np.array(channel_data)  信道记录
        # print('channel_data', channel_data)
        # print('length', len(channel_data))


        # ========== 结果保存 ==========

        # 使用 os.path.join 自动适配平台（Mac用/, Windows用\）
        file_folder = os.path.join('result', 'overlay', f'channel_{num_channel}_su_{num_su}_punish_-2_-10_R')

        # 创建目录
        os.makedirs(file_folder, exist_ok=True)

        # 保存文件时也用 os.path.join
        np.save(os.path.join(file_folder, 'success_history'), success_history)
        np.save(os.path.join(file_folder, 'fail_PU_history'), fail_PU_history)
        np.save(os.path.join(file_folder, 'fail_collision_history'), fail_collision_history)
        np.save(os.path.join(file_folder, 'overall_reward'), overall_reward)
        np.save(os.path.join(file_folder, 'QoS_history'), QoS_history)

        df = pd.DataFrame(success_history)
        df.to_excel(os.path.join(file_folder, 'success_history.xlsx'), index=False)

        df = pd.DataFrame(fail_PU_history)
        df.to_excel(os.path.join(file_folder, 'fail_PU_history.xlsx'), index=False)

        df = pd.DataFrame(fail_collision_history)
        df.to_excel(os.path.join(file_folder, 'fail_collision_history.xlsx'), index=False)

        df = pd.DataFrame(overall_reward)
        df.to_excel(os.path.join(file_folder, 'overall_reward.xlsx'), index=False)

        df = pd.DataFrame(QoS_history)
        df.to_excel(os.path.join(file_folder, 'QoS_history.xlsx'), index=False)

