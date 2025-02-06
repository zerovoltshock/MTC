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
