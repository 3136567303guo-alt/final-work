"""
期末大作业题目2：数值积分与误差分析
内容：梯形/Simpson法阶数验证、高斯积分无限区间处理、蒙特卡洛模拟
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad # 用于作为标准答案验证

# 设置绘图风格和字体
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 基础积分方法实现
# ==========================================
def trapezoidal_rule(func, a, b, n):
    """ 梯形公式 """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    # T = h/2 * (f(x0) + 2*sum(f_mid) + f(xn))
    integral = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return integral

def simpson_rule(func, a, b, n):
    """ Simpson 1/3 公式 (要求 n 为偶数) """
    if n % 2 != 0: raise ValueError("Simpson规则要求 n 必须为偶数")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    # S = h/3 * (f(x0) + 4*sum(odd) + 2*sum(even) + f(xn))
    integral = (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
    return integral

# ==========================================
# 任务1：验证 sin(x) 积分的收敛阶
# ==========================================
print("\n--- 任务1：基本数值积分误差分析 ---")
true_val_1 = 2.0 # int_0^pi sin(x) dx = 2
n_list = [4, 8, 16, 32, 64]
errors_trap = []
errors_simp = []

for n in n_list:
    res_trap = trapezoidal_rule(np.sin, 0, np.pi, n)
    res_simp = simpson_rule(np.sin, 0, np.pi, n)
    
    err_trap = abs(res_trap - true_val_1)
    err_simp = abs(res_simp - true_val_1)
    
    errors_trap.append(err_trap)
    errors_simp.append(err_simp)
    print(f"n={n:2d} | Trap误差: {err_trap:.2e} | Simp误差: {err_simp:.2e}")

# 绘图：双对数坐标图
plt.figure(figsize=(10, 6))
plt.loglog(n_list, errors_trap, 'o-', label='梯形法 (Trapezoidal)')
plt.loglog(n_list, errors_simp, 's-', label='辛普森法 (Simpson)')

# 添加辅助线验证斜率
# 梯形法 O(1/n^2) -> 斜率 -2
ref_trap = [errors_trap[0] * (n_list[0]/n)**2 for n in n_list]
plt.loglog(n_list, ref_trap, 'k--', alpha=0.5, label='参考线 Slope=-2')

# Simpson法 O(1/n^4) -> 斜率 -4
ref_simp = [errors_simp[0] * (n_list[0]/n)**4 for n in n_list]
plt.loglog(n_list, ref_simp, 'r:', alpha=0.5, label='参考线 Slope=-4')

plt.title('任务1: 数值积分误差收敛性分析 (Log-Log Plot)', fontsize=14)
plt.xlabel('分割数 n (log scale)', fontsize=12)
plt.ylabel('绝对误差 (log scale)', fontsize=12)
plt.legend()
plt.grid(True, which="both", ls="-")
plt.tight_layout()
plt.show() # 保存为图1

# ==========================================
# 任务2：高斯积分 (无限区间处理)
# ==========================================
print("\n--- 任务2：高斯积分 (截断法 vs 变换法) ---")
true_val_2 = np.sqrt(np.pi)
print(f"真实值 sqrt(pi) = {true_val_2:.10f}")

# 方法A: 截断法 [-L, L]
def gaussian(x): return np.exp(-x**2)
L_values = [2, 5, 10, 20]
print(">> 方法A: 截断区间法")
for L in L_values:
    # 使用 Simpson 法计算，n取大一点保证精度
    res = simpson_rule(gaussian, -L, L, 1000)
    err = abs(res - true_val_2)
    print(f"L={L:2d} | 积分结果: {res:.10f} | 误差: {err:.2e}")

# 方法B: 变量代换法
# 变换 x = t / sqrt(1-t^2)
# dx = (1-t^2)^(-3/2) dt
# 积分区间变为 (-1, 1)
def transformed_integrand(t):
    # 防止除0错误，加个极小值保护
    # 但更好的方法是缩减积分区间
    with np.errstate(divide='ignore', invalid='ignore'):
        x = t / np.sqrt(1 - t**2)
        dx_dt = (1 - t**2)**(-1.5)
        val = np.exp(-x**2) * dx_dt
    return np.nan_to_num(val) # 处理端点的NaN

# 积分区间取 (-1+eps, 1-eps) 避开奇点
eps = 1e-7
res_trans = simpson_rule(transformed_integrand, -1+eps, 1-eps, 1000)
err_trans = abs(res_trans - true_val_2)
print(f">> 方法B: 变量代换法 (区间映射到[-1,1])")
print(f"积分结果: {res_trans:.10f} | 误差: {err_trans:.2e}")

# ==========================================
# 任务3：蒙特卡洛积分 (Monte Carlo)
# ==========================================
print("\n--- 任务3：蒙特卡洛积分收敛性 ---")
# 目标: int_0^1 e^(-x^2) dx
# 使用 scipy.integrate.quad 获取这一题的精确解作为参考
true_val_3, _ = quad(lambda x: np.exp(-x**2), 0, 1)

N_values = [100, 1000, 10000, 100000, 1000000]
mc_errors = []

print(f"目标真值: {true_val_3:.8f}")
for N in N_values:
    # 1. 生成随机点
    x_rand = np.random.uniform(0, 1, N)
    # 2. 计算函数值
    y_vals = np.exp(-x_rand**2)
    # 3. 均值 * 区间长度(1-0)
    res_mc = np.mean(y_vals) * 1.0
    
    err = abs(res_mc - true_val_3)
    mc_errors.append(err)
    print(f"N={N:<7} | MC结果: {res_mc:.8f} | 误差: {err:.2e}")

# 绘图：蒙特卡洛误差
plt.figure(figsize=(10, 6))
plt.loglog(N_values, mc_errors, 'o-', label='蒙特卡洛误差')

# 理论参考线 1/sqrt(N) -> Slope -0.5
ref_mc = [mc_errors[0] * (N_values[0]/N)**0.5 for N in N_values]
plt.loglog(N_values, ref_mc, 'r--', label='理论参考线 Slope=-0.5')

plt.title('任务3: 蒙特卡洛积分误差分析 (Error vs N)', fontsize=14)
plt.xlabel('采样点数 N (log scale)', fontsize=12)
plt.ylabel('绝对误差 (log scale)', fontsize=12)
plt.legend()
plt.grid(True, which="both", ls="-")
plt.tight_layout()
plt.show() # 保存为图2