import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# ===== 実測データ（仮のデータ） =====
time = np.linspace(0, 1, 10)  # 時間 (秒)
measured_positions = np.array([
    [0.2, 0.1, 0.3], [0.22, 0.12, 0.32], [0.24, 0.14, 0.34],
    [0.26, 0.16, 0.36], [0.28, 0.18, 0.38], [0.30, 0.20, 0.40],
    [0.32, 0.22, 0.42], [0.34, 0.24, 0.44], [0.36, 0.26, 0.46],
    [0.38, 0.28, 0.48]
])  # 各時刻の手首位置 (x, y, z)

# ===== 最小Jerkモデルの軌道生成 =====
def min_jerk_trajectory(start, goal, T, t_vals):
    """
    最小Jerkモデルによる軌道生成
    """
    # 5次多項式の係数を求める
    A = np.array([
        [0, 0, 0, 0, 0, 1],       # P(0) = start
        [T**5, T**4, T**3, T**2, T, 1],  # P(T) = goal
        [0, 0, 0, 0, 1, 0],       # dP/dt(0) = 0
        [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0],  # dP/dt(T) = 0
        [0, 0, 0, 2, 0, 0],       # d²P/dt²(0) = 0
        [20*T**3, 12*T**2, 6*T, 2, 0, 0]  # d²P/dt²(T) = 0
    ])
    
    b = np.array([start, goal, 0, 0, 0, 0])  # 境界条件

    # x, y, z それぞれの軌道を計算
    coeffs = np.array([solve(A, np.array([start[i], goal[i], 0, 0, 0, 0])) for i in range(3)])
    
    # 時間ごとの軌道を計算
    T_mat = np.array([t_vals**5, t_vals**4, t_vals**3, t_vals**2, t_vals, np.ones_like(t_vals)]).T
    trajectory = np.dot(T_mat, coeffs.T)
    
    return trajectory

# 最適軌道の計算
optimal_positions = min_jerk_trajectory(measured_positions[0], measured_positions[-1], 1.0, time)

# ===== 誤差計算 =====
position_error = measured_positions - optimal_positions

# ===== 結果表示 =====
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')

# 実測データ
ax.plot(measured_positions[:, 0], measured_positions[:, 1], measured_positions[:, 2], 'ro-', label="Measured")
# 最小Jerkモデル
ax.plot(optimal_positions[:, 0], optimal_positions[:, 1], optimal_positions[:, 2], 'bo-', label="Minimum Jerk")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
ax.set_title("Hand Trajectory: Measured vs Minimum Jerk")

# 誤差プロット
ax2 = fig.add_subplot(122)
ax2.plot(time, np.linalg.norm(position_error, axis=1), 'k-', label="Error")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Position Error (m)")
ax2.legend()
ax2.set_title("Position Error over Time")

plt.show()

print("位置誤差 (m):", position_error)
