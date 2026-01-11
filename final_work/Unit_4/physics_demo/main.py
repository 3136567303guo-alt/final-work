import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from physics_model import run_simulation, calculate_resonance_curve

st.set_page_config(page_title="物理仿真终极版", layout="wide")

st.title("🧪 深度物理交互：受迫振动与共振 (Plan C)")
st.markdown("### 探索过程：引入周期性驱动力与频域分析")

# --- 侧边栏布局 ---
with st.sidebar:
    st.header("1. 系统属性 (System)")
    m = st.slider("质量 m (kg)", 0.1, 5.0, 1.0, 0.1)
    k = st.slider("劲度 k (N/m)", 1.0, 50.0, 20.0, 1.0)
    c = st.slider("阻尼 c (N·s/m)", 0.0, 5.0, 0.5, 0.1)
    
    # 计算固有频率并在界面展示
    wn = np.sqrt(k/m)
    st.info(f"💡 系统固有频率 $\omega_n$ = {wn:.2f} rad/s")

    st.markdown("---")
    st.header("2. 外部驱动 (Driver)")
    F0 = st.slider("驱动力幅值 F0 (N)", 0.0, 10.0, 5.0, 0.5)
    w_dr = st.slider("驱动频率 $\omega_{dr}$ (rad/s)", 0.0, 15.0, 2.0, 0.1)
    
    # 提示共振点
    if abs(w_dr - wn) < 0.5:
        st.warning("⚠️ 接近共振频率！小心振幅爆炸！")

    st.markdown("---")
    st.header("3. 初始状态")
    x0 = st.number_input("初始位移", value=1.0)
    v0 = st.number_input("初始速度", value=0.0)
    duration = st.slider("时长 (s)", 10.0, 50.0, 30.0)

# --- 计算 ---
t, x, v = run_simulation(m, k, c, x0, v0, F0, w_dr, duration)

# --- 布局：上面放两张图，下面放一张图 ---
row1_col1, row1_col2 = st.columns([2, 1])

# 1. 时域图
with row1_col1:
    st.subheader("时域响应 (Time Domain)")
    fig1, ax1 = plt.subplots(figsize=(8, 3.5))
    ax1.plot(t, x, 'b-', label='Displacement', linewidth=1.5)
    # 画出驱动力的包络参考（缩放以便观察）
    ax1.plot(t, 0.5 * F0 * np.cos(w_dr * t), 'g--', alpha=0.3, label='Driver Signal (Scaled)')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

# 2. 相图
with row1_col2:
    st.subheader("相轨迹 (Phase Space)")
    fig2, ax2 = plt.subplots(figsize=(4, 3.5))
    # 渐变色绘制轨迹，可以看出时间演化
    ax2.plot(x, v, color='purple', linewidth=1)
    # 标记最后的状态
    ax2.plot(x[-1], v[-1], 'ro', label='Current State')
    ax2.set_xlabel("x")
    ax2.set_ylabel("v")
    ax2.axis('equal')
    ax2.grid(True)
    st.pyplot(fig2)

# 3. 频域/共振曲线 (这是 Plan C 的核心亮点)
st.subheader("幅频响应与共振检测 (Frequency Response)")
w_range = np.linspace(0, 15, 200)
A_theory = calculate_resonance_curve(m, k, c, F0, w_range)

fig3, ax3 = plt.subplots(figsize=(10, 3))
# 绘制理论曲线
ax3.plot(w_range, A_theory, 'k-', label='Theoretical Resonance Curve')
# 填充颜色
ax3.fill_between(w_range, A_theory, color='orange', alpha=0.2)
# 标记当前驱动频率的位置
current_amp = calculate_resonance_curve(m, k, c, F0, [w_dr])[0]
ax3.plot(w_dr, current_amp, 'ro', markersize=10, label=f'Current Driver ($\omega_{{dr}}={w_dr}$)')
ax3.axvline(wn, color='blue', linestyle='--', alpha=0.5, label='Natural Freq ($\omega_n$)')

ax3.set_xlabel("Driving Frequency (rad/s)")
ax3.set_ylabel("Steady State Amplitude")
ax3.legend()
ax3.grid(True)
st.pyplot(fig3)