import matplotlib.pyplot as plt
import os

# ================= 路径配置 =================
# 注意：这里的路径是相对于项目根目录的
# 你的数据结构是 result/overlay/channel_8...
# 请确保这个路径字符串完全匹配你截图里的文件夹名
DATA_DIR = "result/overlay/channel_8_su_6_punish_-2_-10_MULTIQ" 

# ================= 绘图配置 =================
# 平滑窗口大小：你跑了1.5万次，建议设小一点，比如200-500
SMOOTH_WINDOW = 1

def setup_plot(title, ylabel, ylim):
    """
    统一的绘图设置函数
    """
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=14)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # 强制设置横坐标范围（为了和师兄的图对比）
    plt.xlim(0, 100000) 
    # 设置纵坐标范围
    plt.ylim(ylim)
    
    plt.grid(True, linestyle='--', alpha=0.6)