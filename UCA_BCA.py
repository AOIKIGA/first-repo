import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持，可根据实际情况调整
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# 数据整理，结构：{码字参数: {方法: {SNR: 对应值}}}
data = {
    "(2, 1, 3)": {
        "UCA": {4: 5.1, 5: 6.18, 6: 7.36},
        "BCA": {4: 5.35, 5: 6.68, 6: 8.34}
    },
    "(3, 1, 3)": {
        "UCA": {4: 4.71, 5: 5.58, 6: 6.41},
        "BCA": {4: 5.55, 5: 6.83, 6: 8.35}
    },
    "(3, 2, 3)": {
        "UCA": {4: 5.5, 5: 6.84, 6: 8.33},
        "BCA": {4: 5.67, 5: 7.1, 6: 9.06}
    }
}

# 方法列表
methods = ["UCA", "BCA"]
# SNR 列表
snrs = [4, 5, 6]
# 码字参数列表，用于横坐标分组
code_params = list(data.keys())

bar_width = 0.35  # 每组内柱子的宽度
# 计算横坐标位置，每个码字参数对应一组（包含多个 SNR ）
x = np.concatenate([np.arange(len(snrs)) + i * (len(snrs) + 1) for i in range(len(code_params))])

fig, ax = plt.subplots(figsize=(14, 6))

# 定义每种方法对应的颜色
method_colors = {
    "UCA": "skyblue",
    "BCA": "lightgreen"
}

# 遍历方法和码字参数绘制柱状图
for method_idx, method in enumerate(methods):
    for param_idx, param in enumerate(code_params):
        # 提取该码字参数、该方法在各 SNR 下的值
        values = [data[param][method][snr] for snr in snrs]
        # 计算该组柱子的横坐标偏移
        offset_x = np.arange(len(snrs)) + param_idx * (len(snrs) + 1)
        ax.bar(
            offset_x + method_idx * bar_width,
            values,
            width=bar_width,
            label=method if param_idx == 0 else "",
            color=method_colors[method]  # 指定每种方法对应的颜色
        )

# 隐藏上、右、下边框，仅保留左侧 y 轴
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# 重新计算 x 轴刻度位置，每个 SNR 对应一个刻度
xticks = []
xticklabels = []
for param_idx, param in enumerate(code_params):
    # 计算每个 SNR 对应柱子的中心位置
    for snr_idx, snr in enumerate(snrs):
        center_x = snr_idx + param_idx * (len(snrs) + 1) + bar_width / 2
        xticks.append(center_x)
        xticklabels.append(snr)
    # 在每组 SNR 之间添加码字参数标签
    if param_idx < len(code_params) :
        xticks.append((xticks[-1] + (param_idx + 1) * (len(snrs) + 1)) / 2)
        xticklabels.append(param)

# 设置横坐标刻度和标签
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, fontsize=20)

# 旋转 x 轴标签以便更好显示

# 仅最左侧显示 y 轴刻度，隐藏其他子图（这里只有一个图，主要是规范设置 ）
# 若有多个子图，可通过循环控制，这里简单处理
ax.yaxis.set_ticks_position('left')

# 设置 y 轴刻度值字号
ax.tick_params(axis='y', labelsize=20)

# 添加 y 轴标签
ax.set_ylabel('-ln(BER)', fontsize=20)
# 添加图例，仅在合适位置显示一次，设置图例字号
ax.legend(fontsize=20)

# 添加网格线（可选，根据需求调整 ）
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()