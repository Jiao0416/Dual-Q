import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 在这里直接定义配置，不再从 config.py 导入 ---
# 1. 修改这里：确保路径正确指向你的数据文件夹
DATA_DIR = "result/overlay/channel_8_su_6_punish_-2_-10_MULTIQ" 

# 2. 修改这里：平滑窗口
SMOOTH_WINDOW = 5


def moving_average(data, window):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def main():
    # 检查路径是否存在
    if not os.path.exists(DATA_DIR):
        print(f"❌ 错误：找不到数据文件夹 -> {DATA_DIR}")
        print("请检查 DATA_DIR 路径是否正确！")
        return

    file_path = os.path.join(DATA_DIR, "success_history.xlsx")
    
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到文件 -> {file_path}")
        return

    print(f"✅ 正在读取数据: {file_path} ...")

    # 读取 Excel (假设没有表头，读第一列)
    try:
        df = pd.read_excel(file_path, header=None)
        raw_data = df.iloc[:, 0].values 
    except Exception as e:
        print(f"❌ 读取 Excel 失败: {e}")
        return

    print(f"📊 数据总量: {len(raw_data)} 条")

    # 平滑处理
    smooth_data = moving_average(raw_data, SMOOTH_WINDOW)
    
    # 生成 x 轴 (平滑后的数据会变短，需要修正 x 轴起点)
    x_smooth = range(SMOOTH_WINDOW - 1, len(raw_data))

    # --- 绘图 ---
    plt.figure(figsize=(10, 6))
    
    # 画出曲线
    plt.plot(x_smooth, smooth_data, label='My Method', color='red', linewidth=2)
    
    # 设置和师兄一样的坐标轴范围
    plt.xlim(0, 100)
    # plt.xlim(0, len(raw_data)) # 自动适应
    # plt.ylim(0.0, 1.0)  # 纵坐标 0 到 1
    
    plt.title("Success Access Rate Comparison", fontsize=14)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Success Rate", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 保存并显示
    output_name = "my_success_rate.png"
    plt.savefig(output_name, dpi=300)
    print(f"🎉 图片已保存: {output_name}")
    plt.show()

if __name__ == "__main__":
    main()