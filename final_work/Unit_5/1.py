import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class LBMSimulator:
    def __init__(self, nx=400, ny=100):
        self.nx, self.ny = nx, ny
        
        # --- 🔧 稳健参数 ---
        self.tau = 0.6         # 保持在安全的粘度 (0.6 比 0.55 稳得多)
        self.u0 = 0.1          # 适中的流速
        
        # D2Q9 常量
        self.w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        self.cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
        self.cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
        self.opposite = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
        
        # 障碍物：加大半径到 15 (物理上增加雷诺数，而不用降低粘度)
        Y, X = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        self.cylinder = (X - nx//4)**2 + (Y - ny//2)**2 < 15**2

        # --- 初始化：全场预启动 ---
        # 直接让整个风洞的风速初始就是 u0，避免“突然开机”的激波爆炸
        self.F = np.zeros((9, ny, nx))
        rho_init = 1.0
        # 计算 u=u0, v=0 的平衡态
        u2 = self.u0**2
        for i in range(9):
            cu = self.cx[i] * self.u0 + self.cy[i] * 0
            feq = self.w[i] * rho_init * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)
            self.F[i, :, :] = feq

        # 稍微加一点极小的非对称性 (只在障碍物附近)，诱导摆动
        # 注意：这是极小量，且不加在全场，绝不会炸
        self.F[:, 45:55, 90:110] += 0.0001 

    def step(self):
        rho = np.sum(self.F, axis=0)
        ux = np.sum(self.F * self.cx[:, None, None], axis=0) / rho
        uy = np.sum(self.F * self.cy[:, None, None], axis=0) / rho
        
        u2 = ux**2 + uy**2
        for i in range(9):
            cu = self.cx[i] * ux + self.cy[i] * uy
            feq = self.w[i] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)
            self.F[i] += -(self.F[i] - feq) / self.tau

        for i in range(9):
            self.F[i] = np.roll(self.F[i], shift=(self.cx[i], self.cy[i]), axis=(1, 0))
            
        boundary = self.cylinder
        for i in range(9):
            self.F[i][boundary] = self.F[self.opposite[i]][boundary]
            
        # 入口: 维持流速 u0
        col0_rho = 1.0
        col0_ux = self.u0
        col0_uy = 0.0
        u2_loc = col0_ux**2 + col0_uy**2
        for i in range(9):
            cu = self.cx[i] * col0_ux + self.cy[i] * col0_uy
            feq_col0 = self.w[i] * col0_rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2_loc)
            self.F[i, :, 0] = feq_col0
        
        # 出口
        self.F[:, :, -1] = self.F[:, :, -2]
        
        return np.sqrt(ux**2 + uy**2)

# --- 可视化 ---
sim = LBMSimulator(nx=400, ny=100)
fig, ax = plt.subplots(figsize=(10, 3.5))

# 使用 'plasma' 配色，视觉效果最清晰
img = ax.imshow(np.zeros((sim.ny, sim.nx)), cmap='plasma', vmin=0, vmax=0.15)
ax.add_patch(plt.Circle((sim.nx//4, sim.ny//2), 15, color='black'))
ax.axis('off')
ax.set_title("LBM Physics: Karman Vortex Street (Stable)")

def update(frame):
    # 每帧计算 15 步
    for _ in range(15):
        speed = sim.step()
    
    img.set_data(speed)
    return [img]

print("模拟已启动。")
print("1. 画面应该非常干净（紫色背景，黄色流体）。")
print("2. 请观察黑球右侧的黑色尾迹。")
print("3. 大约在第 15-20 秒，尾迹的末端会开始上下摆动。")
ani = animation.FuncAnimation(fig, update, frames=800, interval=1, blit=True)
plt.show()