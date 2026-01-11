import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import os
import warnings

# ==========================================
# 0. 救命级配置 (英文模式)
# ==========================================
warnings.filterwarnings("ignore")
current_dir = os.path.dirname(os.path.abspath(__file__))

# 【核心修改】
# 1. 彻底放弃中文字体，回归最安全的默认英文字体 (DejaVu Sans)
# 2. 这样负号 (-) 绝对能显示出来
plt.rcParams.update(plt.rcParamsDefault) 
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False # 强制使用短横线

print("Starting... (English Mode for perfect rendering)")

# ==========================================
# Task 1
# ==========================================
def error_analysis_task1():
    n_list = np.array([4, 8, 16, 32, 64])
    h_list = []
    errors_trap = []
    errors_simp = []

    for n in n_list:
        h = np.pi / n
        h_list.append(h)
        x = np.linspace(0, np.pi, n + 1)
        y = np.sin(x)
        res_trap = (h/2) * (y[0] + 2*np.sum(y[1:-1]) + y[-1])
        res_simp = (h/3) * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]) + y[-1])
        errors_trap.append(abs(res_trap - 2.0))
        errors_simp.append(abs(res_simp - 2.0))

    plt.figure(figsize=(10, 6))
    # 英文标签
    plt.loglog(h_list, errors_trap, 'o-', label='Trapezoidal Rule')
    plt.loglog(h_list, errors_simp, 's-', label='Simpson\'s Rule')
    plt.loglog(h_list, [errors_trap[0]*(h/h_list[0])**2 for h in h_list], 'k--', alpha=0.3, label='Ref Slope=2')
    plt.loglog(h_list, [errors_simp[0]*(h/h_list[0])**4 for h in h_list], 'r--', alpha=0.3, label='Ref Slope=4')
    
    plt.title('Task 1: Integration Error Analysis (Log-Log)')
    plt.xlabel('Step Size h')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    
    save_path = os.path.join(current_dir, 'task2_fig1.png')
    plt.savefig(save_path, dpi=100)
    plt.close()

# ==========================================
# Task 2 (Table Output)
# ==========================================
def task2_gaussian():
    print("\n" + "="*50)
    print("【 Please Screenshot This Table below! 】")
    print("="*50)
    print(f"{'Method / L':<15} | {'Value':<12} | {'Error'}")
    print("-" * 50)
    
    true_val = np.sqrt(np.pi)
    for L in [1, 2, 3, 5]:
        val, _ = integrate.quad(lambda x: np.exp(-x**2), -L, L)
        err = abs(val - true_val)
        print(f"Truncation L={L:<2} | {val:.8f}   | {err:.2e}")
        
    def transformed(t):
        denom = (1 - t**2)**1.5 + 1e-15
        x = t / (np.sqrt(1 - t**2) + 1e-15)
        return np.exp(-x**2) / denom
    val_trans, _ = integrate.quad(transformed, -1, 1)
    
    print("-" * 50)
    print(f"Substitution    | {val_trans:.8f}   | {abs(val_trans - true_val):.2e}")
    print("="*50 + "\n")

# ==========================================
# Task 3
# ==========================================
def task3_monte_carlo():
    N_list = [100, 1000, 10000, 50000]
    errors = []
    for N in N_list:
        x = np.random.uniform(0, 1, N)
        errors.append(abs(np.mean(np.exp(-x**2)) - 0.7468241328))
    
    plt.figure(figsize=(10, 6))
    plt.loglog(N_list, errors, 'o-', label='Monte Carlo Error')
    plt.loglog(N_list, [errors[0]*(N_list[0]/n)**0.5 for n in N_list], 'r--', label='Ref Line 1/sqrt(N)')
    
    plt.title('Task 3: Monte Carlo Convergence')
    plt.xlabel('Number of Samples N')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    
    save_path = os.path.join(current_dir, 'task2_fig2.png')
    plt.savefig(save_path, dpi=100)
    plt.close()

if __name__ == "__main__":
    error_analysis_task1()
    task2_gaussian()
    task3_monte_carlo()
    print("Done! Check your images.")