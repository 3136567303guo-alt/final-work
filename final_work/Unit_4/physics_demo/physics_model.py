import numpy as np
from scipy.integrate import odeint

def system_equations(y, t, m, k, c, F0, w_dr):
    """
    微分方程组: m*a + c*v + k*x = F0 * cos(w_dr * t)
    变型为: a = (F0*cos(w_dr*t) - k*x - c*v) / m
    """
    x, v = y
    
    # 驱动力 (Driving Force)
    F_drive = F0 * np.cos(w_dr * t)
    
    dxdt = v
    dvdt = (F_drive - k * x - c * v) / m
    
    return [dxdt, dvdt]

def run_simulation(m, k, c, x0, v0, F0, w_dr, duration):
    """
    运行受迫振动模拟
    """
    t = np.linspace(0, duration, 1000)
    initial_state = [x0, v0]
    
    # 将新参数传给方程
    solution = odeint(system_equations, initial_state, t, args=(m, k, c, F0, w_dr))
    
    x = solution[:, 0]
    v = solution[:, 1]
    
    return t, x, v

def calculate_resonance_curve(m, k, c, F0, w_range):
    """
    计算理论上的幅频响应曲线 (Resonance Curve)
    用于对比实验值与理论值
    公式: A = F0 / sqrt( (k - m*w^2)^2 + (c*w)^2 )
    """
    amplitudes = []
    for w in w_range:
        denom = np.sqrt((k - m * w**2)**2 + (c * w)**2)
        if denom == 0:
            amplitudes.append(0) # 避免除以零
        else:
            amplitudes.append(F0 / denom)
    return np.array(amplitudes)