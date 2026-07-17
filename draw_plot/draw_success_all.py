import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 配置区 ---
DATA_DIR = "resultpro_2000/50/overlay"
SMOOTH_WINDOW = 2  # 平滑窗口，保持和之前一致

# ⭐ 核心修改点：在这里定义你要对比的文件和标签 ⭐
# 格式：{'file': '文件名', 'label': '图例名称', 'color': '颜色'}
# 你可以根据实际情况修改文件名和颜色
METHODS = [
    {'file': 'channel_8_su_6_punish_-2_-10_MULTIQ/success_history.xlsx', 'label': 'MULTIQ', 'color': 'red'}, 

    {'file': 'channel_8_su_6_all0.7_DualQPlus/success_history.xlsx', 'label': 'DualQPlus-egreedy0.85', 'color': 'black'}, 

    {'file': 'channel_8_su_6_DualQESoftmax_tau_1.0/success_history.xlsx', 'label': 'DualQ-εSoftmax', 'color': 'purple'},

    {'file': 'channel_8_su_6_DualQERandom/success_history.xlsx', 'label': 'DualQ-εRandom', 'color': 'cyan'},

    {'file': 'channel_8_su_6_DualQGreedy/success_history.xlsx', 'label': 'DualQ-Greedy', 'color': 'magenta'},

    # {'file': 'channel_8_su_6_egreedy0.9_DualQPlus/success_history.xlsx', 'label': 'DualQPlus-egreedy0.9', 'color': 'green'}, 

    # {'file': 'channel_8_su_6_softmask0.7_DualQPlus/success_history.xlsx', 'label': 'softmask0.7', 'color': 'green'}, 

    {'file': 'channel_8_su_6_explore_DualQPlus/success_history.xlsx', 'label': 'explore', 'color': 'orange'},

    {'file': 'channel_8_su_6_punish_-2_-10_Q/success_history.xlsx', 'label': 'Qlearning', 'color': 'blue'},   

    {'file': 'channel_8_su_6_QLSTM/success_history.xlsx', 'label': 'QLSTM', 'color': 'brown'}, 

    {'file': 'channel_8_su_6_punish_-2_-10_R/success_history.xlsx',   'label': 'Random',  'color': 'gray'}     
]

def moving_average(data, window):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def main():
    plt.figure(figsize=(12, 7)) # 图片稍微拉大一点，方便看图例

    print(f"🚀 开始绘制对比图...")

    # 根据实际config修改
    TOTAL_REQUESTS_PER_BATCH = 2000 

    # 循环读取并绘制每一种方法
    for method in METHODS:
        file_name = method['file']
        label = method['label']
        color = method['color']
        
        file_path = os.path.join(DATA_DIR, file_name)
        
        if not os.path.exists(file_path):
            print(f"⚠️ 警告：找不到文件 {file_name}，跳过。")
            continue
            
        try:
            # 1. 读取数据
            df = pd.read_excel(file_path, header=None)
            raw_data = df.iloc[:, 0].values
            
            # 2. 平滑处理 次数
            # smooth_data = moving_average(raw_data, SMOOTH_WINDOW)
            # x_smooth = range(SMOOTH_WINDOW - 1, len(raw_data))

            # ⭐⭐⭐ 关键修改：将“次数”转换为“比率” ⭐⭐⭐
            smooth_data = moving_average(raw_data, SMOOTH_WINDOW)
            smooth_rate = smooth_data / TOTAL_REQUESTS_PER_BATCH
            x_smooth = range(SMOOTH_WINDOW - 1, len(raw_data))
            
            # 3. 绘图 (在同一个画布上画线)
            # linewidth=2 让线条稍微粗一点，更清晰
            # 次数
            # plt.plot(x_smooth, smooth_data, label=label, color=color, linewidth=2)
            # 率
            plt.plot(x_smooth, smooth_rate, label=label, color=color, linewidth=2)
            
            print(f"✅ 成功绘制: {label}")
            
        except Exception as e:
            print(f"❌ 读取 {file_name} 失败: {e}")

    # --- 图表美化 ---
    plt.ylim(0.0, 1.0)

    plt.title("Success Rate Comparison", fontsize=16)
    plt.xlabel("Iterations (x200 steps)", fontsize=12)
    plt.ylabel("Success_rate", fontsize=12)
    
    # 显示图例
    plt.legend(fontsize=12, loc='lower right') # loc='lower right' 把图例放在右下角，不挡线
    
    # 显示网格
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 如果你想强制Y轴范围（比如 0-220），取消下面这行的注释
    # plt.ylim(0, 220)

    # 保存和显示
    output_name = "Success_rate.png"
    plt.savefig(output_name, dpi=300)
    print(f"🎉 对比图已保存: {output_name}")
    plt.show()

if __name__ == "__main__":
    main()
