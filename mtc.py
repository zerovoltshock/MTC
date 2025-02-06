import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ===== 実測データ（仮のデータ） =====
time = np.linspace(0, 1, 10)  # 時間 (秒)
measured_positions = np.array([
    [0.2, 0.1, 0.3], [0.22, 0.12, 0.32], [0.24, 0.14, 0.34],
    [0.26, 0.16, 0.36], [0.28, 0.18, 0.38], [0.30, 0.20, 0.40],
    [0.32, 0.22, 0.42], [0.34, 0.24, 0.44], [0.36, 0.26, 0.46],
    [0.38, 0.28, 0.48]
])  # 各時刻の手先位置 (x, y, z)

# ===== アームパラメータ =====
L1, L2, L3 = 0.3, 0.25, 0.2  # 各リンク長
m1, m2, m3 = 2.0, 1.5, 1.0   # 各リンク質量
g = 9.81  # 重力加速度

# ===== 逆運動学 (簡易版) =====
def inverse_kinematics(position):
    x, y, z = position
    q1 = np.arctan2(y, x)  # 肩の回転
    r = np.sqrt(x**2 + y**2)
    d = np.sqrt(r**2 + (z - L1)**2)  # 2D平面での距離
    q2 = np.arccos((L2**2 + d**2 - L3**2) / (2 * L2 * d))  # 肘の角度
    q3 = np.arccos((L2**2 + L3**2 - d**2) / (2 * L2 * L3))  # 手首の角度
    return np.array([q1, q2, q3])

# 実測データを関節角度に変換
measured_angles = np.array([inverse_kinematics(p) for p in measured_positions])

# ===== 最小トルク変化モデルの計算 =====
def cost_function(q_traj):
    """
    目的関数: トルク変化を最小化
    """
    q_traj = q_traj.reshape(-1, 3)  # (N, 3) の形に戻す
    dq = np.gradient(q_traj, axis=0)  # 角速度
    ddq = np.gradient(dq, axis=0)  # 角加速度
    torque_change = np.sum(ddq**2)  # トルク変化の最小化
    return torque_change

# 初期・終端条件
q_start = measured_angles[0]  # 初期角度
q_goal = measured_angles[-1]  # 目標角度

# 最適化の初期値（線形補間）
q_init = np.linspace(q_start, q_goal, len(time)).flatten()

# 最適化実行
result = minimize(cost_function, q_init, method='L-BFGS-B')
optimal_q = result.x.reshape(-1, 3)  # 最適な角度軌道

# ===== 誤差計算 =====
angle_error = measured_angles - optimal_q

# ===== 結果表示 =====
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i, joint in enumerate(["q1 (Shoulder)", "q2 (Elbow)", "q3 (Wrist)"]):
    axs[i].plot(time, np.degrees(measured_angles[:, i]), 'ro-', label="Measured")
    axs[i].plot(time, np.degrees(optimal_q[:, i]), 'bo-', label="MTC")
    axs[i].set_xlabel("Time (s)")
    axs[i].set_ylabel(f"{joint} Angle (deg)")
    axs[i].legend()
    axs[i].set_title(f"{joint}: Measured vs MTC")

plt.show()

print("角度誤差 (deg):", np.degrees(angle_error))
