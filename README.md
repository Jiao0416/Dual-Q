师兄论文：Bayesian-Inference-Assisted Dual-Q Learning for  Dynamic Spectrum Access

系统中有：

* PU：主用户，有授权信道，优先级最高；
* SU：次用户，只能在 PU 空闲时机会式接入；
* 多个信道；
* 多个 SU 同时选择信道；
* 信道状态由马尔可夫过程变化；
* SU 只能通过频谱感知和 ACK 反馈间接了解环境。

目标是让 SU 学会：

* 尽量接入空闲信道；
* 避免与 PU 碰撞；
* 避免与其他 SU 碰撞；
* 提高整体 QoS / 吞吐量；
* 加快学习收敛速度。

这篇论文的核心创新不是“用了 LSTM”本身，而是把 LSTM 预测结果作为贝叶斯后验权重，用来融合成功收益 Q 表和碰撞惩罚 Q 表，从而让 DSA 中的信道选择更快收敛、更少碰撞、更高 QoS。

| 方法          | 类型               | 特点                                            |
| ------------- | ------------------ | ----------------------------------------------- |
| Random Access | 随机接入           | 不学习，随机选信道                              |
| Q-learning    | 表格型 RL          | 单 Q 表，无预测，无正负反馈分离                 |
| DQN           | 深度 Q 网络        | 用 MLP 近似 Q 值，但仍是单估计器                |
| Double DQN    | 深度 RL            | 缓解 Q 值过估计，但仍是单 Q 估计                |
| Dueling DQN   | 深度 RL            | 拆分 value / advantage，但仍不分奖励和惩罚      |
| DRQN          | 循环深度 RL        | 加入 recurrent 结构处理历史信息，但仍是单估计器 |
| DQL           | 论文提出方法       | 表格型 Dual-Q + LSTM 后验融合                   |
| Dual-Q DQN    | 论文提出的深度版本 | 深度网络版 Dual-Q + 后验融合                    |

这篇论文是把 LSTM 与双 Q 学习融合，并用 LSTM 输出构造“贝叶斯后验权重”，动态调整成功收益表和碰撞惩罚表在决策中的占比，从而选择期望收益更高、碰撞风险更低的信道。

我的改进是将探索阶段从均匀随机探索改为 LSTM 预测引导的定向探索。也就是在 exploration 分支中，不再所有信道等概率随机选，而是根据信道预测空闲概率进行加权采样。





| 当前代码 / 训练标志                         | 图例名                  | 含义                   | 利用阶段                                            | 探索阶段                                            |
| ------------------------------------------- | ----------------------- | ---------------------- | --------------------------------------------------- | --------------------------------------------------- |
| `MultipleQ.py`/`flag_MultipleQLearning` | `DualQ`               | 原始 baseline          | 使用 LSTM 预测动态融合双 Q，选择 fused Q 最大的信道 | 均匀随机选择一个信道                                |
| `DualQ_plus.py`/`flag_DualQPlus`        | `The proposed method` | 预测引导的定向探索方法 | 使用 LSTM 预测动态融合双 Q，选择 fused Q 最大的信道 | 根据 LSTM 预测的空闲概率加权随机选择信道            |
| `DualQeRandom.py`/`flag_DualQERandom`   | `DualQ-TopKRandom`    | Top-K 候选集随机探索   | 使用 LSTM 预测动态融合双 Q，选择 fused Q 最大的信道 | 先取 fused Q 排名前 K 的信道，再从 Top-K 中随机选择 |
| `DualQeSoftmax.py`/`flag_DualQESoftmax` | `DualQ-Softmax`       | Softmax 概率探索       | 使用 LSTM 预测动态融合双 Q，选择 fused Q 最大的信道 | 根据 fused Q 的 Softmax 概率分布随机选择信道        |
| `DualQGreedy.py`/`flag_DualQGreedy`     | `DualQ-Greedy`        | 纯贪婪，无探索         | 使用 LSTM 预测动态融合双 Q，选择 fused Q 最大的信道 | 无探索，始终执行利用                                |
| `flag_random`                             | `Random`              | 完全随机接入           | 不利用 Q 表                                         | 所有信道均匀随机选择                                |


文章定位：

现有 Dual-Q 方法主要将 LSTM 预测用于 Q 值融合，但探索阶段仍采用随机探索。由于 DSA 中随机探索会带来碰撞和干扰，本文提出一种预测引导探索策略，将 LSTM 预测的信道空闲概率用于探索阶段的动作采样，从而提高探索效率。



DSA 中 SU 要动态选信道
    ↓
强化学习可以让 SU 通过交互学习接入策略
    ↓
传统 Q-learning 单 Q 表混合成功奖励和碰撞惩罚，收敛慢
    ↓
DualQ + LSTM 方法改进了 Q 值估计
    ↓
但它在探索阶段仍然随机选信道
    ↓
随机探索在 DSA 中代价高，因为可能造成 PU/SU 碰撞
    ↓
本文提出 prediction-guided exploration
    ↓
探索时根据 LSTM 预测空闲概率加权采样
    ↓
实验表明成功率、QoS、碰撞表现更优
