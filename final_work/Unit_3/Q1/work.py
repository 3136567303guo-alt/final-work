import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# ==========================================
# 【新增】解决中文显示和负号显示问题
# ==========================================
# Windows 系统通常使用 SimHei (黑体)
plt.rcParams['font.sans-serif'] = ['SimHei'] 

# Mac 系统如果报错，请注释掉上面那行，改用下面这行：
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 

# 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# ==========================================
# 1. 基础参数设置 (全局变量)
# ==========================================
g = 9.8       # 重力加速度 (m/s^2)
v0 = 20.0     # 初速度 (m/s)
m = 1.0       # 质量 (kg)

# ==========================================
# 2. 核心物理模型函数 (工程化封装)
# ==========================================

def no_drag_trajectory(v0, theta_deg):
    """
    计算无阻力情况下的解析解
    """
    theta_rad = np.radians(theta_deg)
    # 计算飞行总时间 (y回到0的时间) T = 2*v0*sin(theta)/g
    T_flight = 2 * v0 * np.sin(theta_rad) / g
    t = np.linspace(0, T_flight, 200)
    
    # 运动方程
    x = v0 * np.cos(theta_rad) * t
    y = v0 * np.sin(theta_rad) * t - 0.5 * g * t**2
    return t, x, y

def drag_differential_eq(state, t, b, m, g):
    """
    定义空气阻力微分方程组 (用于 odeint 数值求解)
    State向量: [x, vx, y, vy]
    """
    x, vx, y, vy = state
    dxdt = vx
    dydt = vy
    # F_drag = -b * v  => a = - (b/m) * v
    dvxdt = -(b / m) * vx
    dvydt = -g - (b / m) * vy
    return [dxdt, dvxdt, dydt, dvydt]

def solve_with_drag(v0, theta_deg, b_val):
    """
    求解含阻力的运动轨迹
    """
    theta_rad = np.radians(theta_deg)
    # 初始状态 [x, vx, y, vy]
    init_state = [0, v0 * np.cos(theta_rad), 0, v0 * np.sin(theta_rad)]
    
    # 预估一个足够长的时间，后期截断
    t_span = np.linspace(0, 5, 500)
    
    # 数值积分求解
    sol = odeint(drag_differential_eq, init_state, t_span, args=(b_val, m, g))
    
    x = sol[:, 0]
    y = sol[:, 2]
    vx = sol[:, 1]
    vy = sol[:, 3]
    
    # 数据清洗：只保留 y >= 0 的部分 (落地停止)
    idx = np.where(y >= 0)[0]
    if len(idx) == 0: return np.array([]), np.array([]), np.array([])
    
    valid_len = idx[-1]
    return t_span[:valid_len+1], x[:valid_len+1], y[:valid_len+1], vx[:valid_len+1], vy[:valid_len+1]

# ==========================================
# 3. 主程序：绘图与分析
# ==========================================

plt.style.use('seaborn-v0_8-whitegrid') # 这一行必须在最前面！

# --- 【关键修改】字体设置必须放在 style.use 之后！---
# Windows 尝试微软雅黑 (Microsoft YaHei)，如果不行尝试黑体 (SimHei)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# --- 任务1：无阻力多角度模拟 ---
plt.figure(figsize=(10, 6))
angles = [15, 30, 45, 60, 75]
max_range_no_drag = 0

for theta in angles:
    t, x, y = no_drag_trajectory(v0, theta)
    plt.plot(x, y, label=f'θ={theta}°')
    if x[-1] > max_range_no_drag:
        max_range_no_drag = x[-1]

# 标注最大射程
plt.annotate(f'Max Range (45°): {max_range_no_drag:.2f}m', 
             xy=(max_range_no_drag, 0), xytext=(max_range_no_drag-10, 5),
             arrowprops=dict(facecolor='red', shrink=0.05))

plt.title('任务1: 无空气阻力的抛体运动 (Ideal Projectile Motion)')
plt.xlabel('Distance x (m)')
plt.ylabel('Height y (m)')
plt.axhline(0, color='black', lw=1)
plt.legend()
plt.tight_layout()
plt.show()

# --- 任务2：空气阻力影响分析 ---
plt.figure(figsize=(10, 6))
b_values = [0, 0.1, 0.3, 0.5]
theta_fixed = 45

print(f"\n--- 任务2分析：阻力系数对射程的影响 (θ={theta_fixed}°) ---")
for b in b_values:
    t, x, y, _, _ = solve_with_drag(v0, theta_fixed, b)
    if len(x) > 0:
        dist = x[-1]
        print(f"阻力系数 b/m = {b}: 射程 ≈ {dist:.2f} m")
        plt.plot(x, y, label=f'b/m={b}')

plt.title(f'任务2: 不同阻力系数下的轨迹对比 (θ={theta_fixed}°)')
plt.xlabel('Distance x (m)')
plt.ylabel('Height y (m)')
plt.axhline(0, color='black', lw=1)
plt.legend()
plt.tight_layout()
plt.show()

# --- 任务2进阶：寻找有阻力时的最佳发射角 ---
# 假设 b/m = 0.5
b_test = 0.5
best_angle = 0
max_r = 0
print(f"\n--- 进阶分析：寻找 b/m={b_test} 时的最佳发射角 ---")

angle_scan = range(20, 60, 1) # 扫描20到60度
ranges = []
for ang in angle_scan:
    _, x, _, _, _ = solve_with_drag(v0, ang, b_test)
    if len(x) > 0:
        r = x[-1]
        ranges.append(r)
        if r > max_r:
            max_r = r
            best_angle = ang

print(f"计算结果：当 b/m={b_test} 时，最大射程为 {max_r:.2f}m")
print(f"最佳发射角为：{best_angle}° (不再是45°)")

# --- 任务3：能量分析 ---
plt.figure(figsize=(10, 5))
# 选取 b/m = 0.3 进行分析
b_energy = 0.3
t, x, y, vx, vy = solve_with_drag(v0, 45, b_energy)

# 计算能量
v_squared = vx**2 + vy**2
E_kinetic = 0.5 * m * v_squared
E_potential = m * g * y
E_total = E_kinetic + E_potential

plt.plot(t, E_total, label='Total Mechanical Energy', color='red', linewidth=2)
plt.plot(t, E_kinetic, label='Kinetic Energy', linestyle='--', alpha=0.7)
plt.plot(t, E_potential, label='Potential Energy', linestyle=':', alpha=0.7)

plt.title(f'任务3: 机械能随时间变化 (b/m={b_energy})')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()