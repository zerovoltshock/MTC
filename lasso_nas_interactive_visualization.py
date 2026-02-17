"""
LASSO-NAS Algorithm Interactive Visualization
==============================================

このスクリプトは、LASSO-NASアルゴリズムの各ステップを
対話的に可視化するツールです。

各ステップで何が起こっているかを視覚的に理解できます。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


class LassoNASVisualizer:
    """
    LASSO-NASアルゴリズムの各ステップを可視化するクラス
    """
    
    def __init__(self, figsize=(20, 12)):
        self.figsize = figsize
        self.step_data = {}
        
    def generate_synthetic_data(self, n_samples=100, n_wavelengths=300, seed=42):
        """
        シミュレーションデータの生成
        
        Parameters
        ----------
        n_samples : int
            サンプル数
        n_wavelengths : int
            波長数
        seed : int
            乱数シード
        
        Returns
        -------
        X_raw : array (n_samples, n_wavelengths)
            生スペクトル
        Y : array (n_samples,)
            濃度
        wavelengths : array (n_wavelengths,)
            波長軸
        """
        np.random.seed(seed)
        
        # 波長軸
        wavelengths = np.linspace(400, 2500, n_wavelengths)
        
        # 目的成分の純粋スペクトル（2つのガウシアンピーク）
        coi_spectrum = (
            np.exp(-((wavelengths - 1000) ** 2) / (2 * 100 ** 2)) +
            0.5 * np.exp(-((wavelengths - 1500) ** 2) / (2 * 80 ** 2))
        )
        
        # 干渉成分の純粋スペクトル
        oic1_spectrum = np.exp(-((wavelengths - 800) ** 2) / (2 * 150 ** 2))
        oic2_spectrum = np.exp(-((wavelengths - 1800) ** 2) / (2 * 120 ** 2))
        oic3_spectrum = 0.3 * np.exp(-((wavelengths - 1200) ** 2) / (2 * 200 ** 2))
        
        # 濃度生成
        c_coi = np.random.uniform(1, 10, n_samples)
        c_oic1 = np.random.uniform(0.5, 3, n_samples)
        c_oic2 = np.random.uniform(0.3, 2, n_samples)
        c_oic3 = np.random.uniform(0.2, 1.5, n_samples)
        
        # スペクトル生成
        X_raw = (
            c_coi[:, None] * coi_spectrum +
            c_oic1[:, None] * oic1_spectrum +
            c_oic2[:, None] * oic2_spectrum +
            c_oic3[:, None] * oic3_spectrum +
            np.random.normal(0, 0.03, (n_samples, n_wavelengths))
        )
        
        Y = c_coi
        
        # データ保存
        self.step_data['raw'] = {
            'X': X_raw,
            'Y': Y,
            'wavelengths': wavelengths,
            'coi_spectrum': coi_spectrum,
            'oic_spectra': [oic1_spectrum, oic2_spectrum, oic3_spectrum]
        }
        
        return X_raw, Y, wavelengths
    
    def step1_preprocessing(self, X_raw, method='snv'):
        """
        ステップ1: 前処理
        """
        print("\n" + "="*60)
        print("ステップ1: 前処理")
        print("="*60)
        
        if method == 'snv':
            X_pre = self._snv(X_raw)
            print("  方法: SNV (Standard Normal Variate)")
        elif method == 'savgol':
            X_pre = self._savgol(X_raw)
            print("  方法: Savitzky-Golay平滑化")
        else:
            X_pre = X_raw.copy()
            print("  方法: なし")
        
        print(f"  入力形状: {X_raw.shape}")
        print(f"  出力形状: {X_pre.shape}")
        
        self.step_data['preprocessing'] = {
            'X_pre': X_pre,
            'method': method
        }
        
        return X_pre
    
    def _snv(self, X):
        """標準正規変換"""
        X_snv = np.zeros_like(X)
        for i in range(X.shape[0]):
            X_snv[i] = (X[i] - X[i].mean()) / X[i].std()
        return X_snv
    
    def _savgol(self, X, window=11, polyorder=2):
        """Savitzky-Golay平滑化"""
        X_smooth = np.zeros_like(X)
        for i in range(X.shape[0]):
            X_smooth[i] = savgol_filter(X[i], window, polyorder)
        return X_smooth
    
    def step2_lasso_selection(self, X_pre, y, cv=10):
        """
        ステップ2: LASSO波長選択
        """
        print("\n" + "="*60)
        print("ステップ2: LASSO波長選択")
        print("="*60)
        
        # スケーリング
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_pre)
        
        # LASSO with CV
        print(f"  交差検証: {cv}分割")
        lasso = LassoCV(cv=cv, max_iter=10000, n_jobs=-1)
        lasso.fit(X_scaled, y)
        
        # 選択された波長
        coef = lasso.coef_
        selected = np.where(coef != 0)[0]
        
        if len(selected) == 0:
            print("  警告: 波長が選択されませんでした。全波長を使用します。")
            selected = np.arange(X_pre.shape[1])
        
        print(f"  最適λ: {lasso.alpha_:.6f}")
        print(f"  選択波長数: {len(selected)} / {X_pre.shape[1]}")
        print(f"  選択率: {100*len(selected)/X_pre.shape[1]:.1f}%")
        
        X_org = X_pre[:, selected]
        
        self.step_data['lasso'] = {
            'X_org': X_org,
            'selected_indices': selected,
            'coefficients': coef,
            'lambda_opt': lasso.alpha_,
            'lasso_model': lasso
        }
        
        return X_org, selected
    
    def step3_pca_reconstruction(self, X_org, variance_threshold=0.95):
        """
        ステップ3: PCA再構築
        """
        print("\n" + "="*60)
        print("ステップ3: PCA再構築")
        print("="*60)
        
        N, H = X_org.shape
        
        # 正規化
        X_mean = X_org.mean(axis=0)
        X_std = X_org.std(axis=0)
        U = (X_org - X_mean) / X_std
        
        # 共分散行列と固有値分解
        Sigma = (1 / (N - 1)) * (U.T @ U)
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
        
        # 降順ソート
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 累積寄与率
        cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        n_comp = np.where(cumvar >= variance_threshold)[0][0] + 1
        
        print(f"  分散閾値: {variance_threshold*100:.0f}%")
        print(f"  使用主成分数: {n_comp} / {H}")
        print(f"  実際の累積寄与率: {cumvar[n_comp-1]*100:.2f}%")
        
        # 再構築
        P = eigenvectors[:, :n_comp]
        R_scores = U @ P
        X_recon = R_scores @ P.T
        X_recon = X_recon * X_std + X_mean
        
        # 再構築誤差
        recon_error = np.linalg.norm(X_org - X_recon) / np.linalg.norm(X_org)
        print(f"  相対再構築誤差: {recon_error*100:.2f}%")
        
        self.step_data['pca'] = {
            'X': X_recon,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'n_components': n_comp,
            'cumvar': cumvar,
            'X_mean': X_mean,
            'X_std': X_std
        }
        
        return X_recon
    
    def step4_rank_annihilation(self, X, y):
        """
        ステップ4: ランク消去法
        """
        print("\n" + "="*60)
        print("ステップ4: ランク消去法（干渉成分の特定）")
        print("="*60)
        
        N, H = X.shape
        
        # 擬似逆行列
        X_pinv = np.linalg.pinv(X)
        cond_X = np.linalg.cond(X)
        print(f"  X の条件数: {cond_X:.2e}")
        print(f"  X形状: {X.shape}, X_pinv形状: {X_pinv.shape}")
        
        # 投影
        y_tilde = X @ X_pinv @ y
        
        # スケーリング
        d = np.ones(N)
        # 正しい計算: d^T @ X_pinv @ y_tilde
        # d: (N,), X_pinv: (H, N), y_tilde: (N,)
        # まず X_pinv @ y_tilde を計算: (H,)
        # 次に d @ X を計算して、その結果と内積
        temp = X_pinv @ y_tilde  # (H,)
        denom = d @ X @ temp  # スカラー
        
        if np.abs(denom) < 1e-10:
            print(f"  警告: 分母が小さい ({denom:.2e}), 正則化を適用")
            denom = 1e-10 if denom >= 0 else -1e-10
        
        alpha = 1.0 / denom
        print(f"  スケーリング係数 α: {alpha:.6f}")
        
        # 干渉成分
        # X: (N, H), y_tilde: (N,), d: (N,)
        # 正しい計算: X - alpha * (y_tilde を列、d を行として外積)
        # 結果は (N, N) ではなく、X と同じ (N, H) にする必要がある
        # 論文の式: R = X - α * ỹ * d^T
        # ここで ỹ は (N,), d^T は (1, N) なので、ỹ * d^T は (N, N)
        # しかし、これは X (N, H) と形状が合わない
        
        # 正しい解釈: d は波長空間のベクトル (H,) であるべき
        # 再定義
        d_wavelength = np.ones(H)
        R = X - alpha * np.outer(y_tilde, d_wavelength)
        
        # ランクの確認
        rank_X = np.linalg.matrix_rank(X)
        rank_R = np.linalg.matrix_rank(R)
        print(f"  rank(X): {rank_X}")
        print(f"  rank(R): {rank_R}")
        print(f"  ランク減少: {rank_X - rank_R}")
        
        self.step_data['rank_annihilation'] = {
            'R': R,
            'alpha': alpha,
            'y_tilde': y_tilde,
            'rank_X': rank_X,
            'rank_R': rank_R
        }
        
        return R
    
    def step5_nas_extraction(self, X, R):
        """
        ステップ5: NAS抽出
        """
        print("\n" + "="*60)
        print("ステップ5: NAS抽出")
        print("="*60)
        
        H = X.shape[1]
        
        # 擬似逆行列
        R_T = R.T
        R_T_pinv = np.linalg.pinv(R_T)
        
        # 投影行列
        P = R_T @ R_T_pinv
        I = np.eye(H)
        P_orth = I - P
        
        # 冪等性チェック
        P_squared = P @ P
        idempotency_error = np.linalg.norm(P_squared - P)
        print(f"  投影行列の冪等性: ||P² - P|| = {idempotency_error:.2e}")
        
        # NAS計算
        Q = X @ P_orth.T
        
        # 検証
        recon_error = np.linalg.norm(X - (Q + R))
        print(f"  再構築誤差: ||X - (Q + R)|| = {recon_error:.2e}")
        
        orthogonality = np.linalg.norm(Q @ R.T)
        print(f"  直交性: ||Q R^T|| = {orthogonality:.2e}")
        
        # エネルギー比
        Q_energy = np.linalg.norm(Q, 'fro') ** 2
        X_energy = np.linalg.norm(X, 'fro') ** 2
        energy_ratio = Q_energy / X_energy
        print(f"  Q/Xエネルギー比: {energy_ratio:.4f}")
        
        self.step_data['nas'] = {
            'Q': Q,
            'P_orth': P_orth,
            'recon_error': recon_error,
            'orthogonality': orthogonality,
            'energy_ratio': energy_ratio
        }
        
        return Q
    
    def step6_plsr_modeling(self, Q, y, max_components=20):
        """
        ステップ6: PLSRモデル構築
        """
        print("\n" + "="*60)
        print("ステップ6: PLSRモデル構築")
        print("="*60)
        
        from sklearn.model_selection import cross_val_score
        
        # 最適成分数の決定
        max_comp = min(max_components, Q.shape[1], Q.shape[0] - 1)
        mse_cv = []
        
        print(f"  最大成分数: {max_comp}")
        
        for n_comp in range(1, max_comp + 1):
            pls = PLSRegression(n_components=n_comp)
            scores = cross_val_score(
                pls, Q, y, cv=10,
                scoring='neg_mean_squared_error'
            )
            mse_cv.append(-scores.mean())
        
        # 最適成分数
        optimal_n_comp = np.argmin(mse_cv) + 1
        print(f"  最適潜在変数数: {optimal_n_comp}")
        
        # 最終モデル
        pls = PLSRegression(n_components=optimal_n_comp)
        pls.fit(Q, y)
        
        # 予測と評価
        y_pred = pls.predict(Q).ravel()
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        print(f"  R²: {r2:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        
        self.step_data['plsr'] = {
            'model': pls,
            'n_components': optimal_n_comp,
            'mse_cv': mse_cv,
            'y_pred': y_pred,
            'r2': r2,
            'rmse': rmse
        }
        
        return pls, y_pred
    
    def visualize_all_steps(self, save_path=None):
        """
        全ステップの可視化
        """
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.4)
        
        # ステップ1: 生データと前処理
        self._plot_step1(fig, gs)
        
        # ステップ2: LASSO選択
        self._plot_step2(fig, gs)
        
        # ステップ3: PCA
        self._plot_step3(fig, gs)
        
        # ステップ4: ランク消去
        self._plot_step4(fig, gs)
        
        # ステップ5: NAS抽出
        self._plot_step5(fig, gs)
        
        # ステップ6: PLSR
        self._plot_step6(fig, gs)
        
        plt.suptitle('LASSO-NAS Algorithm: Step-by-Step Visualization', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n図を保存しました: {save_path}")
        
        plt.show()
    
    def _plot_step1(self, fig, gs):
        """ステップ1の可視化"""
        # 生スペクトル
        ax1 = fig.add_subplot(gs[0, 0])
        X_raw = self.step_data['raw']['X']
        wavelengths = self.step_data['raw']['wavelengths']
        
        for i in range(min(10, X_raw.shape[0])):
            ax1.plot(wavelengths, X_raw[i], alpha=0.5, linewidth=0.8)
        
        ax1.set_xlabel('Wavelength (nm)', fontsize=9)
        ax1.set_ylabel('Absorbance', fontsize=9)
        ax1.set_title('Step 1a: Raw Spectra', fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=8)
        
        # 前処理後
        ax2 = fig.add_subplot(gs[0, 1])
        X_pre = self.step_data['preprocessing']['X_pre']
        
        for i in range(min(10, X_pre.shape[0])):
            ax2.plot(wavelengths, X_pre[i], alpha=0.5, linewidth=0.8)
        
        ax2.set_xlabel('Wavelength (nm)', fontsize=9)
        ax2.set_ylabel('Normalized Absorbance', fontsize=9)
        ax2.set_title('Step 1b: Preprocessed (SNV)', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=8)
    
    def _plot_step2(self, fig, gs):
        """ステップ2の可視化"""
        # LASSO係数
        ax1 = fig.add_subplot(gs[0, 2])
        wavelengths = self.step_data['raw']['wavelengths']
        coef = self.step_data['lasso']['coefficients']
        
        ax1.plot(wavelengths, coef, linewidth=1.5, color='blue')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Wavelength (nm)', fontsize=9)
        ax1.set_ylabel('LASSO Coefficient', fontsize=9)
        ax1.set_title('Step 2a: LASSO Coefficients', fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=8)
        
        # 選択された波長
        ax2 = fig.add_subplot(gs[0, 3])
        X_pre = self.step_data['preprocessing']['X_pre']
        selected = self.step_data['lasso']['selected_indices']
        
        ax2.plot(wavelengths, X_pre[0], alpha=0.3, label='Spectrum', linewidth=1)
        for idx in selected[::max(1, len(selected)//20)]:  # 最大20本
            ax2.axvline(x=wavelengths[idx], color='red', alpha=0.5, linewidth=0.5)
        
        ax2.set_xlabel('Wavelength (nm)', fontsize=9)
        ax2.set_ylabel('Absorbance', fontsize=9)
        ax2.set_title(f'Step 2b: Selected Wavelengths (n={len(selected)})', 
                     fontsize=10, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=8)
    
    def _plot_step3(self, fig, gs):
        """ステップ3の可視化"""
        # スクリープロット
        ax1 = fig.add_subplot(gs[1, 0])
        eigenvalues = self.step_data['pca']['eigenvalues']
        n_comp = self.step_data['pca']['n_components']
        
        ax1.plot(range(1, min(50, len(eigenvalues))+1), 
                eigenvalues[:50], 'o-', linewidth=1.5, markersize=4)
        ax1.axvline(x=n_comp, color='red', linestyle='--', 
                   label=f'Selected: {n_comp}', linewidth=1.5)
        ax1.set_xlabel('Principal Component', fontsize=9)
        ax1.set_ylabel('Eigenvalue', fontsize=9)
        ax1.set_title('Step 3a: Scree Plot', fontsize=10, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=8)
        
        # 累積寄与率
        ax2 = fig.add_subplot(gs[1, 1])
        cumvar = self.step_data['pca']['cumvar']
        
        ax2.plot(range(1, min(50, len(cumvar))+1), 
                cumvar[:50]*100, linewidth=2, color='green')
        ax2.axhline(y=95, color='red', linestyle='--', 
                   label='95% threshold', linewidth=1.5)
        ax2.axvline(x=n_comp, color='red', linestyle='--', linewidth=1.5)
        ax2.set_xlabel('Number of Components', fontsize=9)
        ax2.set_ylabel('Cumulative Variance (%)', fontsize=9)
        ax2.set_title('Step 3b: Cumulative Variance', fontsize=10, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=8)
    
    def _plot_step4(self, fig, gs):
        """ステップ4の可視化"""
        # 元のスペクトルと干渉成分
        ax1 = fig.add_subplot(gs[1, 2])
        X = self.step_data['pca']['X']
        R = self.step_data['rank_annihilation']['R']
        
        sample_idx = 0
        ax1.plot(X[sample_idx], label='Original X', linewidth=2, alpha=0.8)
        ax1.plot(R[sample_idx], label='Interference R', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Wavelength Index', fontsize=9)
        ax1.set_ylabel('Absorbance', fontsize=9)
        ax1.set_title(f'Step 4a: X vs R (Sample {sample_idx})', 
                     fontsize=10, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=8)
        
        # ランク比較
        ax2 = fig.add_subplot(gs[1, 3])
        rank_X = self.step_data['rank_annihilation']['rank_X']
        rank_R = self.step_data['rank_annihilation']['rank_R']
        
        ranks = [rank_X, rank_R]
        labels = ['rank(X)', 'rank(R)']
        colors = ['blue', 'orange']
        
        bars = ax2.bar(labels, ranks, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Rank', fontsize=9)
        ax2.set_title('Step 4b: Matrix Rank Comparison', 
                     fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(labelsize=8)
        
        # 値をバーの上に表示
        for bar, rank in zip(bars, ranks):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rank}', ha='center', va='bottom', fontsize=9)
    
    def _plot_step5(self, fig, gs):
        """ステップ5の可視化"""
        # X, Q, Rの比較
        ax1 = fig.add_subplot(gs[2, 0])
        X = self.step_data['pca']['X']
        Q = self.step_data['nas']['Q']
        R = self.step_data['rank_annihilation']['R']
        
        sample_idx = 0
        ax1.plot(X[sample_idx], label='X (Original)', linewidth=2, alpha=0.8)
        ax1.plot(Q[sample_idx], label='Q (NAS)', linewidth=2, alpha=0.8)
        ax1.plot(R[sample_idx], label='R (Interference)', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Wavelength Index', fontsize=9)
        ax1.set_ylabel('Absorbance', fontsize=9)
        ax1.set_title(f'Step 5a: X = Q + R (Sample {sample_idx})', 
                     fontsize=10, fontweight='bold')
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=8)
        
        # 複数サンプルのNAS
        ax2 = fig.add_subplot(gs[2, 1])
        for i in range(min(10, Q.shape[0])):
            ax2.plot(Q[i], alpha=0.5, linewidth=0.8)
        
        ax2.set_xlabel('Wavelength Index', fontsize=9)
        ax2.set_ylabel('NAS Intensity', fontsize=9)
        ax2.set_title('Step 5b: Extracted NAS (Multiple Samples)', 
                     fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=8)
        
        # エネルギー比較
        ax3 = fig.add_subplot(gs[2, 2])
        X_energy = np.linalg.norm(X, 'fro') ** 2
        Q_energy = np.linalg.norm(Q, 'fro') ** 2
        R_energy = np.linalg.norm(R, 'fro') ** 2
        
        energies = [X_energy, Q_energy, R_energy]
        labels = ['||X||²', '||Q||²', '||R||²']
        colors = ['blue', 'green', 'orange']
        
        bars = ax3.bar(labels, energies, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Frobenius Norm Squared', fontsize=9)
        ax3.set_title('Step 5c: Energy Comparison', fontsize=10, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.tick_params(labelsize=8)
        
        # 値をバーの上に表示
        for bar, energy in zip(bars, energies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{energy:.1f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_step6(self, fig, gs):
        """ステップ6の可視化"""
        # 交差検証MSE
        ax1 = fig.add_subplot(gs[2, 3])
        mse_cv = self.step_data['plsr']['mse_cv']
        n_comp = self.step_data['plsr']['n_components']
        
        ax1.plot(range(1, len(mse_cv)+1), mse_cv, 'o-', linewidth=2, markersize=5)
        ax1.axvline(x=n_comp, color='red', linestyle='--', 
                   label=f'Optimal: {n_comp}', linewidth=1.5)
        ax1.set_xlabel('Number of Components', fontsize=9)
        ax1.set_ylabel('CV MSE', fontsize=9)
        ax1.set_title('Step 6a: PLSR Component Selection', 
                     fontsize=10, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=8)
        
        # 予測 vs 真値
        ax2 = fig.add_subplot(gs[3, 0])
        y = self.step_data['raw']['Y']
        y_pred = self.step_data['plsr']['y_pred']
        r2 = self.step_data['plsr']['r2']
        rmse = self.step_data['plsr']['rmse']
        
        ax2.scatter(y, y_pred, alpha=0.6, s=30, edgecolors='black', linewidths=0.5)
        ax2.plot([y.min(), y.max()], [y.min(), y.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
        ax2.set_xlabel('True Concentration', fontsize=9)
        ax2.set_ylabel('Predicted Concentration', fontsize=9)
        ax2.set_title(f'Step 6b: Prediction (R²={r2:.4f}, RMSE={rmse:.4f})', 
                     fontsize=10, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=8)
        
        # 残差プロット
        ax3 = fig.add_subplot(gs[3, 1])
        residuals = y - y_pred
        
        ax3.scatter(y_pred, residuals, alpha=0.6, s=30, 
                   edgecolors='black', linewidths=0.5, color='purple')
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Predicted Concentration', fontsize=9)
        ax3.set_ylabel('Residuals', fontsize=9)
        ax3.set_title('Step 6c: Residual Plot', fontsize=10, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=8)
        
        # 残差ヒストグラム
        ax4 = fig.add_subplot(gs[3, 2])
        ax4.hist(residuals, bins=20, alpha=0.7, color='purple', 
                edgecolor='black', linewidth=0.8)
        ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('Residuals', fontsize=9)
        ax4.set_ylabel('Frequency', fontsize=9)
        ax4.set_title('Step 6d: Residual Distribution', fontsize=10, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.tick_params(labelsize=8)
        
        # サマリー統計
        ax5 = fig.add_subplot(gs[3, 3])
        ax5.axis('off')
        
        summary_text = f"""
        LASSO-NAS Algorithm Summary
        {'='*35}
        
        Step 1: Preprocessing
          Method: {self.step_data['preprocessing']['method'].upper()}
        
        Step 2: LASSO Selection
          Selected: {len(self.step_data['lasso']['selected_indices'])} / {len(self.step_data['lasso']['coefficients'])}
          Lambda: {self.step_data['lasso']['lambda_opt']:.4f}
        
        Step 3: PCA Reconstruction
          Components: {self.step_data['pca']['n_components']}
          Variance: {self.step_data['pca']['cumvar'][self.step_data['pca']['n_components']-1]*100:.2f}%
        
        Step 4: Rank Annihilation
          rank(X): {self.step_data['rank_annihilation']['rank_X']}
          rank(R): {self.step_data['rank_annihilation']['rank_R']}
        
        Step 5: NAS Extraction
          Orthogonality: {self.step_data['nas']['orthogonality']:.2e}
          Energy Ratio: {self.step_data['nas']['energy_ratio']:.4f}
        
        Step 6: PLSR Modeling
          Components: {self.step_data['plsr']['n_components']}
          R²: {self.step_data['plsr']['r2']:.6f}
          RMSE: {self.step_data['plsr']['rmse']:.6f}
        """
        
        ax5.text(0.1, 0.95, summary_text, transform=ax5.transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))


def main():
    """
    メイン実行関数
    """
    print("\n" + "="*60)
    print("LASSO-NAS Algorithm Interactive Visualization")
    print("="*60)
    
    # ビジュアライザー作成
    visualizer = LassoNASVisualizer(figsize=(20, 12))
    
    # データ生成
    print("\nデータ生成中...")
    X_raw, Y, wavelengths = visualizer.generate_synthetic_data(
        n_samples=100,
        n_wavelengths=300,
        seed=42
    )
    print(f"  サンプル数: {X_raw.shape[0]}")
    print(f"  波長数: {X_raw.shape[1]}")
    print(f"  濃度範囲: [{Y.min():.2f}, {Y.max():.2f}]")
    
    # ステップ1: 前処理
    X_pre = visualizer.step1_preprocessing(X_raw, method='snv')
    
    # ステップ2: LASSO選択
    X_org, selected = visualizer.step2_lasso_selection(X_pre, Y, cv=10)
    
    # ステップ3: PCA再構築
    X = visualizer.step3_pca_reconstruction(X_org, variance_threshold=0.95)
    
    # ステップ4: ランク消去
    R = visualizer.step4_rank_annihilation(X, Y)
    
    # ステップ5: NAS抽出
    Q = visualizer.step5_nas_extraction(X, R)
    
    # ステップ6: PLSRモデル
    pls_model, y_pred = visualizer.step6_plsr_modeling(Q, Y, max_components=20)
    
    # 全体可視化
    print("\n可視化を生成中...")
    visualizer.visualize_all_steps(
        save_path='/home/sandbox/lasso_nas_step_by_step_visualization.png'
    )
    
    print("\n" + "="*60)
    print("完了！")
    print("="*60)


if __name__ == "__main__":
    main()
