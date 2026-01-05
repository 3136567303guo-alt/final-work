import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False    # 负号显示

# ===================== 任务1：方波的傅里叶展开 =====================
def square_wave(x):
    """定义周期2π的方波函数"""
    x = x % (2 * np.pi)  # 周期化
    return 1 if 0 < x < np.pi else -1

def square_fourier_coeff(n):
    """计算方波傅里叶系数 b_n (a_n=0)"""
    if n % 2 == 0:
        return 0.0
    else:
        return 4 / (np.pi * n)

def square_fourier_series(x, N):
    """方波前N项傅里叶级数展开"""
    series = 0.0
    for n in range(1, N+1):
        bn = square_fourier_coeff(n)
        series += bn * np.sin(n * x)
    return series

# 1.1 验证傅里叶系数（打印前10项）
print("=== 方波傅里叶系数验证 ===")
for n in range(1, 11):
    an = 0.0  # 理论值a_n=0
    bn = square_fourier_coeff(n)
    print(f"n={n}: a_n={an:.4f}, b_n={bn:.4f}")

# 1.2 绘制原函数与不同N项傅里叶级数对比
x = np.linspace(-np.pi, 3*np.pi, 2000)  # 扩展区间观察周期性
y_original = np.array([square_wave(xi) for xi in x])

N_list = [3, 5, 11, 51]
plt.figure(figsize=(12, 8))
plt.plot(x, y_original, 'k-', label='原方波', linewidth=2)

colors = ['r', 'g', 'b', 'm']
for i, N in enumerate(N_list):
    y_series = square_fourier_series(x, N)
    plt.plot(x, y_series, colors[i], label=f'N={N}项傅里叶级数', alpha=0.7)

plt.title('方波的傅里叶级数展开（吉布斯现象）', fontsize=14)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(np.pi, color='gray', linestyle='--', alpha=0.5)
plt.ylim(-1.5, 1.5)  # 放大观察吉布斯现象的过冲
plt.show()

# ===================== 任务2：三角波的傅里叶展开 =====================
def triangle_wave(x):
    """定义周期2π的三角波函数 g(x)=|x|, -π<x<π"""
    x = x % (2 * np.pi)  # 周期化
    if x > np.pi:
        x = 2 * np.pi - x
    return x

def triangle_fourier_series(x, N):
    """三角波傅里叶级数展开（推导得：a0=π, an=2*(-1)^n/n² (n≠0), bn=0）"""
    series = np.pi / 2  # a0/2
    for n in range(1, N+1):
        an = 2 * ((-1)**n) / (n**2)
        series += an * np.cos(n * x)
    return series

# 2.1 绘制三角波收敛过程
x = np.linspace(-2*np.pi, 2*np.pi, 2000)
y_original_tri = np.array([triangle_wave(xi) for xi in x])

N_list_tri = [1, 3, 5, 10]
plt.figure(figsize=(12, 8))
plt.plot(x, y_original_tri, 'k-', label='原三角波', linewidth=2)

colors = ['r', 'g', 'b', 'm']
for i, N in enumerate(N_list_tri):
    y_series_tri = triangle_fourier_series(x, N)
    plt.plot(x, y_series_tri, colors[i], label=f'N={N}项傅里叶级数', alpha=0.7)

plt.title('三角波的傅里叶级数展开（收敛过程）', fontsize=14)
plt.xlabel('x')
plt.ylabel('g(x)')
plt.legend()
plt.grid(True)
plt.ylim(-0.5, np.pi + 0.5)
plt.show()

# 2.2 打印收敛分析（文字说明）
print("\n=== 三角波比方波收敛更快的原因 ===")
print("1. 方波存在跳变不连续，傅里叶系数衰减速率为 1/n（仅奇数项非零）；")
print("2. 三角波连续且一阶导数连续（仅二阶导数跳变），傅里叶系数衰减速率为 1/n²；")
print("3. 系数衰减越快，级数收敛越快，因此三角波比方波收敛更快。")

# ===================== 任务3：简单信号合成与FFT =====================
def composite_signal(t):
    """生成复合信号 s(t) = sin(2π·3t) + 0.5sin(2π·7t) + 0.3sin(2π·11t)"""
    return np.sin(2 * np.pi * 3 * t) + 0.5 * np.sin(2 * np.pi * 7 * t) + 0.3 * np.sin(2 * np.pi * 11 * t)

# 3.1 绘制时域波形
t = np.linspace(0, 2, 2048, endpoint=False)  # 2048点保证频率分辨率
y = composite_signal(t)

plt.figure(figsize=(12, 8))
plt.subplot(2,1,1)
plt.plot(t, y)
plt.title('复合信号时域波形', fontsize=14)
plt.xlabel('t (s)')
plt.ylabel('s(t)')
plt.grid(True)

# 3.2 FFT频谱分析
fs = len(t) / 2  # 采样频率：总点数 / 总时长（t∈[0,2]）
y_fft = np.fft.fft(y)
freq = np.fft.fftfreq(len(t), 1/fs)  # 频率轴
amplitude = np.abs(y_fft) / len(t) * 2  # 幅度归一化（直流分量除外）
amplitude[0] /= 2  # 直流分量单独处理

# 只显示正频率部分
mask = freq >= 0
freq_pos = freq[mask]
amp_pos = amplitude[mask]

plt.subplot(2,1,2)
# 修正：移除use_line_collection参数，适配新版matplotlib
plt.stem(freq_pos, amp_pos, basefmt='k-')
plt.title('复合信号FFT频谱', fontsize=14)
plt.xlabel('频率 (Hz)')
plt.ylabel('幅度')
plt.xlim(0, 15)  # 聚焦有效频率范围
plt.grid(True)
plt.tight_layout()
plt.show()