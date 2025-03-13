import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# 時間基底発生器（TBG）の速度モデル
def velocity_model(t, alpha, ts, tf, beta1, beta2):
    """ TBGに基づく速度プロファイルを生成 """
    # 正規化時間変数
    gamma = (np.math.gamma(1 - beta1) * np.math.gamma(1 - beta2)) / (tf * np.math.gamma(2 - (beta1 + beta2)))
    xi = (t - ts) / tf  # 正規化時間
    xi = np.clip(xi, 0, 1)  # 範囲外の値を制限
    
    # 速度モデル
    velocity = alpha * gamma * xi**beta1 * (1 - xi)**beta2
    return velocity

# 残差関数（最適化のための誤差関数）
def residuals(params, t_data, v_data):
    """ 測定データとモデルの誤差を計算 """
    alpha, ts, tf, beta1, beta2 = params
    v_model = velocity_model(t_data, alpha, ts, tf, beta1, beta2)
    return v_data - v_model

# パラメーター推定関数
def estimate_parameters(t_data, v_data):
    """ 最適なパラメータを推定 """
    # 初期推定値 [alpha, ts, tf, beta1, beta2]
    initial_params = [np.max(v_data), t_data[0], t_data[-1] - t_data[0], 0.5, 0.5]
    
    # パラメータの境界条件
    bounds = (
        (0, None),     # alpha: 0以上
        (-1, 1),       # ts: -1 <= ts <= 1（開始時間の変動を考慮）
        (0.1, 2),      # tf: 0.1 <= tf <= 2（運動時間の制約）
        (0.1, 0.9),    # beta1: 0.1 <= beta1 <= 0.9（非線形性を考慮）
        (0.1, 0.9)     # beta2: 0.1 <= beta2 <= 0.9（非線形性を考慮）
    )
    
    # 非線形最小二乗法で最適化
    result = opt.least_squares(residuals, initial_params, bounds=bounds, args=(t_data, v_data), method='trf')
    
    return result.x  # 最適パラメータを返す

# サンプルデータ生成
def generate_sample_data():
    """ TBGモデルに従う仮想的な速度データを生成 """
    t_data = np.linspace(0, 1, 100)  # 時間データ（0〜1秒）
    true_params = [0.3, 0, 1, 0.6, 0.4]  # 真のパラメータ
    v_data = velocity_model(t_data, *true_params) + np.random.normal(0, 0.01, len(t_data))  # ノイズ追加
    return t_data, v_data, true_params

# メイン処理
if __name__ == "__main__":
    # 仮想的な速度データを生成
    t_data, v_data, true_params = generate_sample_data()
    
    # パラメーター推定
    estimated_params = estimate_parameters(t_data, v_data)
    
    # 結果表示
    print("True Parameters (Ground Truth):", true_params)
    print("Estimated Parameters:", estimated_params)
    
    # 推定したパラメータを用いてモデルの速度を計算
    v_estimated = velocity_model(t_data, *estimated_params)
    
    # 速度プロファイルをプロット
    plt.figure(figsize=(8, 5))
    plt.scatter(t_data, v_data, color='blue', label="Measured Data", alpha=0.5)
    plt.plot(t_data, v_estimated, color='red', label="Estimated Model", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.title("Velocity Profile Fitting")
    plt.legend()
    plt.grid()
    plt.show()
