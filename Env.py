"""
认知无线电中的overlay模式环境仿真：TVT的基础上加上实际的位置坐标和实际信道容量模型，
PU马尔可夫信道的设计
用户信道感知
信道状态更新
用户接入和反馈动作设计
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
import matplotlib.pyplot as plt
import copy


class Environment:
    # 初始化
    def __init__(self, num_channel, num_su, num_pu):
        self.num_pu = num_pu
        self.num_channel = num_channel
        self.num_su = num_su

        self.sense_error_prob_max = 0.1  # 感知错误的最大概率，默认0.2，此时感知错误概率为0，即完美感知

        # self.R = 5.2  # 单用户接入速率
        self.f2 = -2  # SU冲突惩罚
        self.f1 = -10  # PU，之所以惩罚值如此大，因为在overlay模式下用户希望完全不干扰PU的活动，越大的惩罚表示尽快退出PU占用的时隙
        self.QoS = 0  # 实际接入网络容量QoS（随迭代次数更新）

        # self.n_actions = num_channel + 1  # 动作空间（选择信道或者不选择信道）
        self.n_actions = num_channel  # 动作空间（信道选择）
        self.n_features = num_channel
        self.sense_error_prob = 0  # 感知错误率

        # Initialize the Markov channels
        self.build_markov_channels()

        # Initialize the locations of SUs and PUs
        self._build_location()

        # SU的信道容量不在相同，会根据实际的位置坐标计算，每个SU有自己的信道容量
        # Set the noise (mW)
        self.Noise = 1 * np.float_power(10, -8)  # 噪声功率设置10^(-8)mW
        # Set the carrier frequency (5 GHz)
        self.fc = 5  # 载波频率设置
        # Set the K in channel gain
        self.K = 8  # 信道增益K因子，Rician 衰落模型中 LOS 分量与非 LOS 分量的比例
        # Set the power of PU and SU (mW)
        self.SU_power = 20  # 发射功率


        # Initialize SINR
        self.render_SINR()  # 初始信噪比的计算
        


        # 构建PU占用信道模型
        # 这是典型的 时变信道建模。PU的行为不是固定的，而是像“忽开忽关”的电视信号，SU必须动态感知并避开
    def build_markov_channels(self):  # 初始化马尔可夫信道状态
        # Initialize channel state (uniform distribution)
        # 0: Inactive state (primary user is not using) 0表示信道未被PU占用
        # 1: Active state (primary user is using) 1表示信道被PU占用

        # 授权信道：前n_pu个信道，PU活动采用马尔可夫模型随机产生状态（1: PU占用，0: PU空闲）
        licensed_channels = self.num_pu
        # 非授权信道：剩余信道，假设始终无PU（状态固定为0，即空闲）
        unlicensed_channels = self.num_channel - self.num_pu

        # 随机生成每个信道的初始状态，状态0表示空闲，状态1表示被占用
        licensed_state = np.random.choice(2, licensed_channels)  # 初始化授权信道的状态，输出为1*num_pu
        unlicensed_state = np.zeros(unlicensed_channels, dtype=int)  # 初始化非授权信道状态，全0空闲，输出为1*(num_channel-num_pu)
        # 将两部分合并，前半部分为授权信道，后半部分为非授权信道
        self.channel_state = np.concatenate((licensed_state, unlicensed_state))

        # 信道转移概率 stayGood保持空闲；stayBad保持占有
        # 这次我们希望每个PU的状态转移概率不要相同，而且我们认为用户在使用信道时要么持续使用，要么持续空闲
        self.stayGood_prob = np.random.uniform(0.8, 1, self.num_pu)
        self.stayBad_prob = np.random.uniform(0.8, 1, self.num_pu)
        self.goodToBad_prob = 1 - self.stayGood_prob
        self.badToGood_prob = 1 - self.stayBad_prob
        # 每个授权信道都有独立的马尔可夫转移概率，保持当前状态的概率在0.8到1.0之间随机分配，这意味着状态相对稳定但仍有动态变化
        

        
        # 构建用户物理位置
        # 初始化PU和SU的地理位置，模拟无线网络中用户设备的物理分布，构建网络拓扑
        # 位置决定路径损耗和干扰强度。SU离PU越近，越容易干扰PU；SU之间距离近，则容易互相干扰。
    def _build_location(self):  # 初始化用户位置
        # ---------------- PU位置 ----------------
        # PU发射端位置：x、y坐标，随机生成，随机分布模拟现实中PU设备的位置不确定性（只针对授权信道）
        self.PU_TX_x = np.random.uniform(0, 150, self.num_pu)
        self.PU_TX_y = np.random.uniform(0, 150, self.num_pu)
        # PU接收机（基站）：固定在中心位置，例如(75, 75)
        base_station_x = 75
        base_station_y = 75
        self.PU_RX_x = np.full(self.num_pu, base_station_x)
        self.PU_RX_y = np.full(self.num_pu, base_station_y)
        # 为了方便计算，为每个PU都"分配"一个接收端位置，但实际上所有位置都是相同的
        # self.PU_RX_x = np.random.uniform(0, 150, self.n_channel)
        # self.PU_RX_y = np.random.uniform(0, 150, self.n_channel)

        # ---------------- SU位置 ----------------
        # SU接收机（AP）：固定位置，位于基站附近（例如(80,80)）
        ap_x = 80
        ap_y = 80
        self.SU_RX_x = np.full(self.num_su, ap_x)
        self.SU_RX_y = np.full(self.num_su, ap_y)
        # self.SU_TX_x = np.random.uniform(0+40, 150-40, self.n_su)  # 确保在[40,110]区域内，不要离边界太近
        # self.SU_TX_y = np.random.uniform(0+40, 150-40, self.n_su)

        # 初始化收发机的距离，在20-40之内（20-40米是典型的室内或短距离通信范围）
        self.SU_d = np.random.uniform(20, 40, self.num_su)

        # 初始化SU发送端的位置，让SU设备围绕AP呈圆形均匀分布
        SU_theda = 2 * np.pi * np.random.uniform(0, 1, self.num_su)  # 随机生成一个角度
        # 极坐标转直角坐标系，极坐标(距离,角度) → 直角坐标(x,y)
        SU_dx = self.SU_d * np.cos(SU_theda)  # 计算偏移量
        SU_dy = self.SU_d * np.sin(SU_theda)
        # 从AP位置出发，加上偏移量，就是发射端坐标：
        self.SU_TX_x = ap_x + SU_dx  
        self.SU_TX_y = ap_y + SU_dy

        # 计算SU接收机对多个PU影响的物理距离
        self.SU_RX_PU_TX_d = np.zeros((self.num_su, self.num_pu))
        for k in range(self.num_su):
            for l in range(self.num_pu):
                # 平方和 = x差平方 + y差平方
                # np.sqrt(平方和): 计算平方根，得到欧氏距离
                self.SU_RX_PU_TX_d[k][l] = np.sqrt(
                    np.float_power(self.SU_RX_x[k] - self.PU_TX_x[l], 2) + np.float_power(
                        self.SU_RX_y[k] - self.PU_TX_y[l], 2))
        
        # 计算SU接收机对多个SU影响的物理距离、欧氏距离
        self.SU_RX_SU_TX_d = np.zeros((self.num_su, self.num_su))
        for k1 in range(self.num_su):
            for k2 in range(self.num_su):
                self.SU_RX_SU_TX_d[k1][k2] = np.sqrt(
                    np.float_power(self.SU_RX_x[k1] - self.SU_TX_x[k2], 2) + np.float_power(
                        self.SU_RX_y[k1] - self.SU_TX_y[k2], 2))

        # Plot the locations  绘制PU和SU 的位置，可视化仿真场景
        # plt.plot(self.PU_TX_x, self.PU_TX_y, 'ro', label='PU_TX')
        # plt.plot(self.PU_RX_x, self.PU_RX_y, 'r^', label='PU_RX (Base station)')
        plt.plot(self.SU_TX_x, self.SU_TX_y, 'bs', label='SU_TX')
        plt.plot(self.SU_RX_x, self.SU_RX_y, 'b^', label='SU_RX (AP)')
        # 设置横纵轴范围
        plt.xlim(0, 150)
        plt.ylim(0, 150)
        # 自动寻找最优位置放置图例
        plt.legend(loc='best')
        # 自动调整布局
        # plt.tight_layout()
        # plt.legend(loc='lower right')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.show()

    
    
    #  信道感知（感知信道是否被PU占用）
        """
        模拟SU对信道的感知能力
        当前设置为完美感知（无错误）
        返回：每个SU感知到的信道状态（0=空闲, 1=占用）
        """
    def sense(self):
        tmp_dice = np.random.uniform(0, 1, size=(self.num_su, self.num_channel))  #  创建num_su行 × num_channel列的矩阵
        error_index = tmp_dice < self.sense_error_prob  # True: 感知错误（1）, False: 感知正确（0）

        # Get the sensing result 感知出错，error为1，状态转换。感知正确时：返回实际状态；感知错误时：返回相反状态
        # 若self.sense_error_prob = 0，则无感知出错率，感知结果即实际结果

        self.sensing_result = self.channel_state * (1 - error_index) + (1 - self.channel_state) * error_index

        return self.sensing_result
   

    # 更新授权信道状态（马尔可夫转移）
    def render(self): 
        # 保持原状态的概率
        stay_prob = self.channel_state[:self.num_pu] * self.stayBad_prob + (1 - self.channel_state[:self.num_pu]) * self.stayGood_prob 
        # 如果当前占用 → 保持概率 = stayBad_prob；如果当前空闲 → 保持概率 = stayGood_prob

        # 每个授权信道掷一个骰子
        # 如果骰子 < 保持概率 → 保持当前状态(stay_index=True)；如果骰子 ≥ 保持概率 → 改变状态(stay_index=False)   
        tmp_dice = np.random.uniform(0, 1, self.num_pu)  # 只更新pu占用信道
        stay_index = tmp_dice < stay_prob  # True: 保持（1）, False: 改变（0）

        # 更新pu所在信道的占用情况
        # 上一时刻状态 → 结合保持概率 → 生成随机数 → 判断是否改变 → 得到新状态
        self.channel_state[:self.num_pu] = self.channel_state[:self.num_pu]*stay_index + (1-self.channel_state[:self.num_pu])*(1-stay_index)
        


    # SU用户接入信道和奖励计算，该接入是一个时隙所有SU的接入结果
    def access(self, action):  
        """
        初始化统计变量
        """
        # action = [1, num_channel]: access the selected channel
        # action = 0: do not access the channel
        # 修改 action = [0, num_channel-1]: 表示选择接入num_channel个信道
        self.success = 0 # 成功接入的SU数量
        self.fail_PU = 0 # 与PU碰撞的SU数量
        self.fail_collision = 0 # SU间碰撞的数量

        self.reward = np.zeros(self.num_su)  # 初始化SU奖励为0
        self.access_su = np.zeros(self.num_su).astype(np.int32)  # 标记哪些SU可以享受QoS奖励（不与PU碰撞）衡量决策质量
        self.access_act = np.zeros(self.num_su).astype(np.int32)  # 标记哪些SU实际成功接入，为了表示ACK 衡量执行结果（是否成功接入）
        self.QoS_su = np.zeros(self.num_su)  # 每个SU接入信道时会获取的信道容量，或根据实际坐标的不同有所差异
        self.reward_type = np.zeros(self.num_su)  # 奖励类型（0=成功，1=碰撞）

        # 跟underlay模式不同的是，SU无法同时接入一个信道，所以只需要计算每个用户SU的信道容量即可
        """
        SINR: 信噪比，衡量信号质量
        self.H2[k, action[k]]: 第k个SU到AP在所选信道上的信道增益
        np.log2(1 + SINR): 香农公式，计算理论最大传输速率 
        """
        for k in range(self.num_su):
            SINR = self.H2[k, action[k]] * self.SU_power / self.Noise # action是一个数组，包含所有SU选择的信道编号，action[k] 表示第k个SU选择的信道编号
            # self.reward[k] = np.log2(1 + SINR)  # 返回每个SU的奖励（若成功接入） 此处的T（SINR gap）未设定
            self.QoS_su[k] = np.log2(1 + SINR)  # 用户实际接入的信道容量

        for k in range(self.num_su):
            
            # 情况1:选择的信道空闲
            if self.channel_state[action[k]] == 0:  # 用户k选择的信道未被PU占用(第K个用户动作选择第action[k]个信道，对应信道列表第action[k]个信道状态)
                self.access_su[k] = 1  # 只要不与PU碰撞，奖励就可以包含QoS，
                if len(np.where(action == action[k])[0]) == 1:  # 只有用户k选择该信道(实际接入)
                    # successful transmission
                    self.success = self.success + 1  # 成功传输，成功传输次数加1
                    self.access_act[k] = 1  # 实际接入记录
                    self.reward_type[k] = 0

            # 情况2：SU间碰撞
                else:  # SU碰撞，实际未接入）
                    # collision with SU
                    self.fail_collision = self.fail_collision + 1  # 否则，用户间碰撞，冲突加1
                    # self.reward[k] = self.R + self.f2  # 奖励即传输速率加惩罚f2
                    self.reward[k] = self.f2
                    self.QoS_su[k] = 0  # 与SU碰撞导致实际接入的信道容量为0
                    self.reward_type[k] = 1

            # 情况3：与PU碰撞
            else:  # 信道被PU占用
                # collision with PU
                self.fail_PU = self.fail_PU + 1  # 与PU碰撞，冲突加1
                self.reward[k] = self.f1  # PU碰撞奖励
                self.QoS_su[k] = 0  # 与PU碰撞，接入信道容量为0
                self.reward_type[k] = 1
                # 如果还有其他SU也选了这个信道 → 还有SU间冲突
                if len(np.where(action == action[k])[0]) > 1:
                    # collision with SU
                    self.fail_collision = self.fail_collision + 1

        # 最终奖励计算
        # 总QoS = 所有SU成功接入的速率之和
        self.QoS = np.sum(self.QoS_su)
        # 最终奖励 = 基础奖励 + QoS收益（仅对成功接入者）
        self.reward = self.reward + self.QoS * self.access_su
        return self.reward, self.QoS, self.access_act, self.reward_type


    
    # 计算信道增益（H²）
    def render_SINR(self):  # SINR计算，计算SU得实际信道增益
        """
        计算SU到其AP的信道增益（考虑路径损耗 + 莱斯衰落）
        使用莱斯K因子模型，模拟视距（LOS）和非视距（NLOS）混合环境
        """
        # Calculate the channel gain 对一维的距离数组进行扩展，变为[n_SU,n_channel]
        SU_d = copy.deepcopy(np.reshape(self.SU_d, (-1, 1)))
        for n in range(self.num_channel-1):
            SU_d = np.hstack((SU_d, np.reshape(self.SU_d, (-1, 1))))
        
        # 路径损耗模型（dB形式转换为线性）
        SU_sigma2 = np.float_power(10, -((41+22.7*np.log10(SU_d)+20*np.log10(self.fc/5))/10))  # 计算SU到接收端的信道衰落因子

        # 生成复高斯随机变量（用于NLOS分量）
        CN_real = np.random.normal(0, 1, size=(self.num_su, self.num_channel))  # 生成高斯分布的随机变量实部
        CN_imag = np.random.normal(0, 1, size=(self.num_su, self.num_channel))  # 生成高斯分布的随机变量虚部

        # 随机相位（用于LOS分量）
        theda = np.random.uniform(0, 1, size=(self.num_su, self.num_channel))  # 均匀分布的随机相位，用于构造莱斯信道

        # 莱斯信道模型：H = sqrt(K/(K+1))*LOS + sqrt(1/(K+1)/2)*NLOS
        H = np.sqrt(self.K/(self.K+1)*SU_sigma2)*np.exp(1j*2*np.pi*theda) + np.sqrt(1/(self.K+1)*SU_sigma2/2)*(CN_real + 1j*CN_imag)

        self.H2 = np.float_power(np.absolute(H), 2)  # |H|²

        # 距离 → 路径损耗 → + 直射分量 → + 多径分量 → 信道增益H2
        #                      ↑              ↑
        #                    K因子控制      随机波动

        # 距离衰减：信号随距离减弱
        # 频率影响：不同频率衰减不同
        # 多径效应：信号经过多条路径到达
        # 莱斯衰落：直射路径+随机散射的综合效果

        # 最终输出的self.H2矩阵包含了每个SU在每个信道上的信道增益平方，为后续的SINR计算提供基础


