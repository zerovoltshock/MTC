import numpy as np
import pandas as pd

# 設定
sampling_rate = 100  # Hz
duration = 5  # 秒
num_samples = sampling_rate * duration  # サンプル数
time = np.linspace(0, duration, num_samples)  # 時間配列

# 軌道生成関数 (最小Jerk軌道に近い滑らかな動き)
def generate_smooth_trajectory(start, end, t):
    T = t[-1]  # 終了時間
    tau = t / T
    position = start + (end - start) * (10*tau**3 - 15*tau**4 + 6*tau**5)
    return position

# 各関節の開始・終了位置
shoulder_start, shoulder_end = np.array([0.0, 0.3, 1.5]), np.array([0.0, 0.35, 1.55])
elbow_start, elbow_end = np.array([0.1, 0.2, 1.2]), np.array([0.15, 0.25, 1.25])
wrist_start, wrist_end = np.array([0.2, 0.1, 0.9]), np.array([0.3, 0.15, 1.0])

# 軌道データ生成
shoulder_traj = np.array([generate_smooth_trajectory(shoulder_start[i], shoulder_end[i], time) for i in range(3)]).T
elbow_traj = np.array([generate_smooth_trajectory(elbow_start[i], elbow_end[i], time) for i in range(3)]).T
wrist_traj = np.array([generate_smooth_trajectory(wrist_start[i], wrist_end[i], time) for i in range(3)]).T

# データフレーム作成
df = pd.DataFrame({
    "time": time,
    "shoulder_x": shoulder_traj[:, 0], "shoulder_y": shoulder_traj[:, 1], "shoulder_z": shoulder_traj[:, 2],
    "elbow_x": elbow_traj[:, 0], "elbow_y": elbow_traj[:, 1], "elbow_z": elbow_traj[:, 2],
    "wrist_x": wrist_traj[:, 0], "wrist_y": wrist_traj[:, 1], "wrist_z": wrist_traj[:, 2]
})

# CSVに保存
csv_filename = "reaching_trajectory.csv"
df.to_csv(csv_filename, index=False)

print(f"サンプルデータを {csv_filename} に保存しました。")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# CSVデータの読み込み
csv_filename = "reaching_trajectory.csv"
df = pd.read_csv(csv_filename)

# 時間データ
time = df["time"].values

# 測定された手首の軌道
wrist_measured = df[["wrist_x", "wrist_y", "wrist_z"]].values

# --- 最小トルクモデルによる理想軌道の生成 ---
# 最小トルク軌道を求めるための関数
def min_torque_trajectory(t, start, end):
    T = t[-1]
    tau = t / T
    position = start + (end - start) * (6*tau**5 - 15*tau**4 + 10*tau**3)
    return position

# 手首の開始・終了位置
wrist_start = wrist_measured[0]
wrist_end = wrist_measured[-1]

# 最小トルク軌道の計算
wrist_mtm = np.array([min_torque_trajectory(time, wrist_start[i], wrist_end[i]) for i in range(3)]).T

# --- プロットによる比較 ---
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

labels = ['X', 'Y', 'Z']
for i in range(3):
    ax[i].plot(time, wrist_measured[:, i], label="Measured", linestyle="dashed", color="blue")
    ax[i].plot(time, wrist_mtm[:, i], label="Minimum Torque Model", color="red")
    ax[i].set_ylabel(f"Position ({labels[i]}) [m]")
    ax[i].legend()

ax[-1].set_xlabel("Time [s]")
plt.suptitle("Wrist Trajectory Comparison: Measured vs Minimum Torque Model")
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 2関節モデルのパラメータ（仮定）
L1 = 0.5  # 上腕の長さ [m]
L2 = 0.5  # 前腕の長さ [m]

# 時間データ
time = np.linspace(0, 1, 1000)  # 1秒間の軌道、1000ポイント

# 最小トルクモデルによる関節角度の計算
def min_torque_trajectory(t, theta_start, theta_end):
    tau = t
    return theta_start + (theta_end - theta_start) * (6*tau**5 - 15*tau**4 + 10*tau**3)

# 肩と肘の初期・最終角度（仮定）
theta1_start = 0  # 肩の初期角度 [rad]
theta1_end = np.pi / 4  # 肩の最終角度 [rad]
theta2_start = 0  # 肘の初期角度 [rad]
theta2_end = np.pi / 4  # 肘の最終角度 [rad]

# 肩と肘の理想的な軌道
theta1_trajectory = min_torque_trajectory(time, theta1_start, theta1_end)
theta2_trajectory = min_torque_trajectory(time, theta2_start, theta2_end)

# 手首の位置（順運動学）
x_wrist = L1 * np.cos(theta1_trajectory) + L2 * np.cos(theta1_trajectory + theta2_trajectory)
y_wrist = L1 * np.sin(theta1_trajectory) + L2 * np.sin(theta1_trajectory + theta2_trajectory)

# プロット
plt.figure(figsize=(10, 8))

# 手首のX-Y軌道
plt.subplot(2, 1, 1)
plt.plot(x_wrist, y_wrist, label="Ideal Wrist Trajectory", color="red")
plt.title("Ideal Wrist Trajectory (2-Joint Model)")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.legend()

# 各関節角度の推移
plt.subplot(2, 1, 2)
plt.plot(time, theta1_trajectory, label="Shoulder Angle (θ1)", color="blue")
plt.plot(time, theta2_trajectory, label="Elbow Angle (θ2)", color="green")
plt.title("Joint Angles (2-Joint Model)")
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.legend()

plt.tight_layout()
plt.show()
