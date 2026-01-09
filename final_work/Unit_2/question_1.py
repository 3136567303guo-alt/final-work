import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# 设置绘图风格，支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def calculate_square_fourier(x, N_terms):
    """
    计算方波的傅里叶级数近似
    f(x) = 4/pi * sum(sin(nx)/n) for n=1, 3, 5...
    """
    result = np.zeros_like(x)
    # 只取奇数项: 1, 3, 5 ...
    # 比如 N=3 时，k取 1, 3
    for n in range(1, 2 * N_terms + 2, 2):
        if n > N_terms: break
        bn = 4 / (n * np.pi)
        result += bn * np.sin(n * x)
    return result

def calculate_triangle_fourier(x, N_terms):
    """
    计算三角波的傅里叶级数近似
    g(x) = pi/2 - 4/pi * sum(cos(nx)/n^2) for n=1, 3, 5...
    注意：这是基于区间[-pi, pi]的推导结果修正。
    根据前文推导: a0/2 = pi/2. an = -4/(n^2*pi) for odd n.
    """
    result = np.full_like(x, np.pi / 2) # a0/2 项
    for n in range(1, 2 * N_terms + 2, 2):
        if n > N_terms: break
        an = -4 / ((n**2) * np.pi)
        result += an * np.cos(n * x)
    return result

# ==========================================
# 任务 1: 方波分析
# ==========================================
x_square = np.linspace(0, 2 * np.pi, 1000)
# 原始方波定义
y_square_true = np.where(x_square < np.pi, 1, -1)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(x_square, y_square_true, 'k-', linewidth=2, label='原函数 (Original)')

# 绘制不同 N 的近似
n_list_sq = [3, 5, 11, 51]
colors = ['r', 'g', 'b', 'orange']

for N, color in zip(n_list_sq, colors):
    y_approx = calculate_square_fourier(x_square, N)
    plt.plot(x_square, y_approx, color=color, linewidth=1, label=f'N={N}')

plt.title('任务1: 方波的傅里叶级数逼近 (Gibbs现象观察)')
plt.legend(loc='upper right', fontsize='small')
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('f(x)')

# ==========================================
# 任务 2: 三角波分析
# ==========================================
x_tri = np.linspace(-np.pi, np.pi, 1000)
y_tri_true = np.abs(x_tri)

plt.subplot(2, 2, 2)
plt.plot(x_tri, y_tri_true, 'k-', linewidth=2, label='原函数 |x|')

n_list_tri = [1, 3, 5, 10]
for N, color in zip(n_list_tri, colors):
    y_approx = calculate_triangle_fourier(x_tri, N)
    plt.plot(x_tri, y_approx, color=color, linewidth=1, linestyle='--', label=f'N={N}')

plt.title('任务2: 三角波的傅里叶级数逼近 (收敛速度对比)')
plt.legend(loc='lower center', fontsize='small')
plt.grid(True, alpha=0.3)
plt.xlabel('x')

# ==========================================
# 任务 3: 信号合成与 FFT
# ==========================================
# 设定采样参数
T = 2.0  # 总时长 2秒
fs = 1000 # 采样率 1000Hz (足够高以避免混叠)
N_samples = int(T * fs)
t = np.linspace(0, T, N_samples, endpoint=False)

# 合成信号
# s(t) = sin(2pi*3t) + 0.5sin(2pi*7t) + 0.3sin(2pi*11t)
s_t = np.sin(2 * np.pi * 3 * t) + \
      0.5 * np.sin(2 * np.pi * 7 * t) + \
      0.3 * np.sin(2 * np.pi * 11 * t)

# 3.1 时域波形
plt.subplot(2, 2, 3)
plt.plot(t, s_t)
plt.title('任务3: 合成信号时域波形 s(t)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(0, 2)
plt.grid(True, alpha=0.3)

# 3.2 频域分析 (FFT)
yf = fft(s_t)
xf = fftfreq(N_samples, 1 / fs)

# 只取正频率部分
positive_idx = xf >= 0
xf_positive = xf[positive_idx]
# 幅度归一化：除以N/2
yf_magnitude = 2.0 / N_samples * np.abs(yf[positive_idx])

plt.subplot(2, 2, 4)
plt.plot(xf_positive, yf_magnitude, 'r-')
plt.title('任务3: 信号频谱图 (FFT)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 15) # 限制显示范围以便看清 3, 7, 11 Hz
plt.xticks(np.arange(0, 16, 1)) # 设置x轴刻度
plt.grid(True, alpha=0.3)

# 标注峰值
for freq in [3, 7, 11]:
    idx = np.argmin(np.abs(xf_positive - freq))
    amp = yf_magnitude[idx]
    plt.text(freq, amp, f'{freq}Hz\n{amp:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 打印验证结果
print("=== 任务3 频谱验证 ===")
print("预期频率: 3Hz (幅度1.0), 7Hz (幅度0.5), 11Hz (幅度0.3)")
print("FFT分析结果:")
for freq in [3, 7, 11]:
    idx = np.argmin(np.abs(xf_positive - freq))
    print(f"频率: {xf_positive[idx]:.2f} Hz, 测量幅度: {yf_magnitude[idx]:.4f}")