# 课程大作业过程文档：题目1 - 傅里叶级数与波形分析



## 一、 问题分析与思考 (Thinking Process)

拿到这道题时，我首先拆解了任务的三个核心部分：
1.  **方波**：这是一个典型的非连续周期信号，我知道它会有“吉布斯现象”，但我不仅要算出来，还要用 Python 画出来验证它。
2.  **三角波**：相比方波，它没有突变，是连续的。直觉告诉我它的频谱衰减应该更快，但我需要通过数学推导和代码绘图来双重验证这一点。
3.  **信号合成与FFT**：这部分是从时域到频域的转换。我的难点在于 Python 的 FFT 库函数怎么用，以及如何把计算出来的数组对应到真实的物理频率（Hz）上。

---

## 二、 详细的 AI 交互与尝试记录 (AI Interaction Log)

为了确保过程的准确性，并解决我在编程和数学上的困惑，我与 AI 进行了深度的多轮交互。以下是我解决本题的完整对话记录。

### 1. 理论推导阶段：积分与系数的困惑

**我问：**
> “我要计算周期为 $2\pi$ 的方波的傅里叶系数。我知道 $a_0$ 是 0，但是 $a_n$ 和 $b_n$ 我总是积错。你能列出详细的积分步骤吗？尤其是 $b_n$，为什么只有奇数项？”

**AI 回答：**
> AI 帮我进行了分部积分推导：
> 1.  **$a_n$**：由于方波（移轴后）是奇函数，而 $\cos(nx)$ 是偶函数，奇 $\times$ 偶 = 奇，对称区间积分为 0。
> 2.  **$b_n$**：计算 $\int_{0}^{\pi} 1 \cdot \sin(nx) dx$ 和 $\int_{\pi}^{2\pi} -1 \cdot \sin(nx) dx$。
> 3.  关键点在于 $\cos(n\pi) = (-1)^n$。
>    *   当 $n$ 为偶数时，$(1 - 1) = 0$。
>    *   当 $n$ 为奇数时，$(1 - (-1)) = 2$。
>    *   最终得出结论：$b_n = \frac{4}{n\pi}$（仅奇数项）。

**我问：**
> “那三角波呢？它是偶函数，$b_n$ 应该是 0 对吧？$a_n$ 的积分里有 $x \cdot \cos(nx)$，这个怎么积？”

**AI 回答：**
> “没错，三角波是偶函数，$b_n=0$。对于 $a_n$，需要使用**分部积分法**：$\int u dv = uv - \int v du$。
> 令 $u=x, dv=\cos(nx)dx$。
> 推导结果显示系数与 $1/n^2$ 成正比，这也是它收敛快的原因。”

---

### 2. 编程实现阶段：从报错到运行

**我问：**
> “我现在要用 Python 画图。我想定义一个函数来算方波的级数和。我写了一个 `for` 循环，但是画出来的图好像只有正弦波，不像方波，为什么？”

**AI 回答：**
> “我检查了你的代码，发现你在循环里写的是 `y = b_n * np.sin(...)`。
> **错误原因**：你每次循环都把 `y` 覆盖了，所以最后只画出了最后一项的正弦波。
> **修正建议**：应该是累加，即 `y += b_n * np.sin(...)`，并且要在循环开始前初始化 `y` 为全零数组。”

**我问：**
> “我想把方波、三角波、合成波形、频谱图这四个图画在一张大图里，也就是 2x2 的排版。还有，我的图表标题想写中文，但是显示出来的全是方框 `□□□`，怎么解决？”

**AI 回答：**
> “1. 使用 `plt.subplot(2, 2, i)` 来控制子图位置。
> 2. 中文乱码是因为 Matplotlib 默认字体不支持中文。请在代码开头加上这两行：
>    ```python
>    plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
>    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
>    ```”

---

### 3. FFT 分析阶段：攻克最难的关卡

这一步我卡了很久，数据一直对不上。

**我问：**
> “我用 `scipy.fft.fft` 对合成信号做了变换，绘图后发现两个问题：
> 1. 横坐标是 0 到 2000，题目里的频率不是 3Hz, 7Hz 吗？
> 2. 纵坐标的幅度有 1000 多，题目公式里的幅度系数明明是 1, 0.5, 0.3 啊。我是不是算错了？”

**AI 回答：**
> “你没有算错，但是缺乏后处理步骤：
> 1. **横坐标问题**：FFT 返回的是对应的数组索引。你需要用 `fftfreq` 函数，结合采样率 $f_s$ 来计算真实的频率轴。
> 2. **幅度归一化**：FFT 的计算结果是信号强度的累加。要得到物理幅度，需要除以采样点数的一半 ($N/2$)。
> **尝试修改代码**：
> `xf = fftfreq(N, 1/fs)`
> `yf_magnitude = 2.0/N * np.abs(yf)`”

**我问：**
> “修改后，我在 3Hz 处看到了峰值。但是图上还有负频率的部分，看着很乱，我能不能只显示正频率？”

**AI 回答：**
> “可以。实数信号的 FFT 频谱是关于 0Hz 对称的。你可以通过切片操作 `xf > 0` 只保留正半轴的数据进行绘图，这样图表会更清晰。”

---

## 三、 尝试与失败的总结 (Trial & Error)

在得出最终完美结果之前，我经历了几次失败的尝试，正是这些尝试让我加深了理解：

1.  **方波绘图失败**：
    *   *尝试*：直接对 $n=1,2,3,4,5$ 求和。
    *   *结果*：波形很奇怪。
    *   *原因*：忘记了推导结论中“偶数项为0”。修正代码逻辑，增加 `step=2` 或判断 `if n % 2 != 0` 后解决。

2.  **FFT 频率对齐失败**：
    *   *尝试*：直接 `plt.plot(fft_result)`。
    *   *结果*：无法读出频率值。
    *   *反思*：深刻理解了采样率 $f_s$ 与频率分辨率的关系，FFT 不是魔法，需要正确的坐标变换。

---

## 四、 结果分析与物理意义 (Analysis)

### 1. 吉布斯现象 (Gibbs Phenomenon) 的观察
在任务 1 的方波图中，我特别放大了跳变点（$x=0, \pi$）附近的波形。
*   **现象描述**：即使当我取 $N=51$ 这么多项时，在垂直跳变的地方，波形并没有平滑过渡，而是出现了一个尖锐的“过冲”（Overshoot），然后迅速衰减震荡。
*   **物理意义**：这说明用连续的函数（正弦波）去逼近一个不连续的函数（方波），在断点处由于收敛的不均匀性，必然存在误差。这个过冲幅度大约是跳变高度的 9%，且不随 N 增大而消失。

### 2. 收敛速度：方波 vs 三角波
*   **现象**：任务 2 的图中，三角波仅仅用到 $N=3$ 时，就已经非常接近原函数了；而方波用到 $N=51$ 还有明显震荡。
*   **数学解释**：
    *   方波系数 $b_n \propto 1/n$（衰减慢，高频能量大）。
    *   三角波系数 $a_n \propto 1/n^2$（衰减快，高频能量极小）。
*   **结论**：信号越平滑（导数连续性越好），其傅里叶系数衰减越快，高频分量越少。

### 3. 频谱分析的意义
在任务 3 中，时域波形看起来是乱糟糟的起伏，肉眼很难看出包含什么成分。
但是一旦转到频域（右下角的图），立刻出现了三根清晰的柱子：
*   3Hz (幅度 1.0)
*   7Hz (幅度 0.5)
*   11Hz (幅度 0.3)
这完美对应了公式 $s(t)=\sin(2\pi\cdot3t)+0.5\sin(2\pi\cdot7t)+0.3\sin(2\pi\cdot11t)$。
这让我明白了**频谱分析**的威力：它能透过现象（复杂的波形）看本质（构成波形的基波成分）。

---

## 五、 最终代码 (Final Solution)
### 附录：核心算法代码实现

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# ==========================================
# 设置绘图风格，支持中文显示
# Engineering Capability: 解决 Matplotlib 中文乱码问题
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def calculate_square_fourier(x, N_terms):
    """
    计算方波的傅里叶级数近似
    理论推导: f(x) = 4/pi * sum(sin(nx)/n) for n=1, 3, 5... (奇数项)
    Input: x (数组), N_terms (级数项数)
    Output: 近似波形数组
    """
    result = np.zeros_like(x)
    # 优化: step=2 确保只遍历奇数 n (1, 3, 5...)，避免无效循环
    for n in range(1, 2 * N_terms + 2, 2):
        if n > N_terms: break
        bn = 4 / (n * np.pi)
        result += bn * np.sin(n * x)
    return result

def calculate_triangle_fourier(x, N_terms):
    """
    计算三角波的傅里叶级数近似
    理论推导: a0/2 = pi/2. an = -4/(n^2*pi) for odd n.
    """
    result = np.full_like(x, np.pi / 2) # 初始项 a0/2
    for n in range(1, 2 * N_terms + 2, 2):
        if n > N_terms: break
        an = -4 / ((n**2) * np.pi)
        result += an * np.cos(n * x)
    return result

# ==========================================
# Main Execution Block
# ==========================================

# 1. 准备数据
x_square = np.linspace(0, 2 * np.pi, 1000)
y_square_true = np.where(x_square < np.pi, 1, -1) # 理想方波
x_tri = np.linspace(-np.pi, np.pi, 1000)
y_tri_true = np.abs(x_tri) # 理想三角波

# 2. 初始化画布 (2x2 Layout)
plt.figure(figsize=(14, 10))

# --- Plot 1: Square Wave ---
plt.subplot(2, 2, 1)
plt.plot(x_square, y_square_true, 'k-', linewidth=2, label='原函数 (Original)')
n_list_sq = [3, 5, 11, 51]
colors = ['r', 'g', 'b', 'orange']
for N, color in zip(n_list_sq, colors):
    y_approx = calculate_square_fourier(x_square, N)
    plt.plot(x_square, y_approx, color=color, linewidth=1, label=f'N={N}')
plt.title('任务1: 方波 - 吉布斯现象观察')
plt.legend(fontsize='small')
plt.grid(True, alpha=0.3)

# --- Plot 2: Triangle Wave ---
plt.subplot(2, 2, 2)
plt.plot(x_tri, y_tri_true, 'k-', linewidth=2, label='原函数 |x|')
n_list_tri = [1, 3, 5, 10]
for N, color in zip(n_list_tri, colors):
    y_approx = calculate_triangle_fourier(x_tri, N)
    plt.plot(x_tri, y_approx, color=color, linewidth=1, linestyle='--', label=f'N={N}')
plt.title('任务2: 三角波 - 快速收敛对比')
plt.legend(loc='lower center', fontsize='small')
plt.grid(True, alpha=0.3)

# --- Plot 3: Signal Synthesis (Time Domain) ---
T = 2.0; fs = 1000; N_samples = int(T * fs)
t = np.linspace(0, T, N_samples, endpoint=False)
s_t = np.sin(2*np.pi*3*t) + 0.5*np.sin(2*np.pi*7*t) + 0.3*np.sin(2*np.pi*11*t)

plt.subplot(2, 2, 3)
plt.plot(t, s_t)
plt.title('任务3: 合成信号时域波形 s(t)')
plt.xlim(0, 2); plt.grid(True, alpha=0.3)
plt.xlabel('Time (s)')

# --- Plot 4: FFT Analysis (Freq Domain) ---
yf = fft(s_t)
xf = fftfreq(N_samples, 1 / fs)
# 只取正频率
pos_idx = xf >= 0
xf_pos = xf[pos_idx]
yf_mag = 2.0 / N_samples * np.abs(yf[pos_idx]) # 归一化幅度

plt.subplot(2, 2, 4)
plt.plot(xf_pos, yf_mag, 'r-')
plt.title('任务3: 信号频谱图 (FFT)')
plt.xlim(0, 15); plt.xticks(np.arange(0, 16, 1))
plt.xlabel('Frequency (Hz)'); plt.grid(True, alpha=0.3)

# 自动标注峰值
for freq in [3, 7, 11]:
    idx = np.argmin(np.abs(xf_pos - freq))
    plt.text(freq, yf_mag[idx], f'{freq}Hz\n{yf_mag[idx]:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

## 六、 运行结果展示 (Result Visualization)
![alt text](./Figure_1.png)
