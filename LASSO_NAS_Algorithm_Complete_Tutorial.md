# LASSO-NAS特徴抽出アルゴリズム完全解説
## Feature Extraction From Spectroscopy Using LASSO and Net Analyte Signal

**論文**: "Feature Extraction From Spectroscopy Using LASSO and Net Analyte Signal"  
**作成日**: 2026年2月17日

---

## 📑 目次

1. [論文の概要と目的](#論文の概要と目的)
2. [NASの役割と重要性](#nasの役割と重要性)
3. [アルゴリズム全体像](#アルゴリズム全体像)
4. [ステップバイステップ詳細解説](#ステップバイステップ詳細解説)
5. [数学的定式化](#数学的定式化)
6. [Python完全実装](#python完全実装)
7. [実例による実演](#実例による実演)
8. [古典的NASとの比較](#古典的nasとの比較)
9. [よくある質問](#よくある質問)

---

## 1. 論文の概要と目的

### 1.1 研究の背景

スペクトル分析において、以下の課題が存在します：

1. **高次元性**: 数百〜数千の波長点（変数）
2. **多重共線性**: 隣接波長間の強い相関
3. **干渉成分**: 目的分析物以外の成分（OICs: Other Interfering Components）
4. **ノイズ**: 測定誤差、環境変動

### 1.2 提案手法の目的

**LASSO + NAS** の組み合わせにより：

1. **次元削減**: LASSOによる重要波長の選択
2. **干渉除去**: NASによるOICsの抑制
3. **モデル簡素化**: より解釈しやすく、ロバストなモデル
4. **予測精度向上**: ノイズと干渉の両方を除去

### 1.3 手法の革新性

従来のNAS手法との違い：

| 特徴 | 従来のNAS | LASSO-NAS |
|------|-----------|-----------|
| 波長選択 | なし（全波長使用） | LASSO事前選択 |
| 計算安定性 | 逆行列計算で不安定 | 選択された波長で安定 |
| ノイズ除去 | NASのみ | LASSO + NAS の二段階 |
| 解釈性 | 中程度 | 高い（重要波長が明確） |

---

## 2. NASの役割と重要性

### 2.1 スペクトルの分解

測定されるスペクトル **x_i** は3つの成分に分解できます：

```
x_i = q_i + r_i + ε_i
```

ここで：
- **q_i**: Net Analyte Signal（目的成分のシグナル）
- **r_i**: 干渉成分（OICs）のシグナル
- **ε_i**: 測定ノイズ

### 2.2 行列表現

サンプル全体では：

```
X = Q + R + E
```

- **X**: 元のスペクトル行列 (N × H)
  - N: サンプル数
  - H: 波長点数
- **Q**: NAS行列（目的成分のみ）(N × H)
- **R**: 干渉成分行列 (N × H)
- **E**: ノイズ行列 (N × H)

### 2.3 NASの物理的意味

**Q行列の各要素 q_ij は**：
- サンプル i における
- 波長 j での
- **目的分析物（COI）に特有のシグナル**
- 干渉成分とノイズを除去した純粋な応答

### 2.4 NASの利点

1. **選択性の向上**: 干渉成分の影響を排除
2. **感度の向上**: ノイズを抑制し、シグナル/ノイズ比を改善
3. **検出限界の低下**: より低濃度の検出が可能
4. **モデルの解釈性**: 目的成分に関連する波長が明確

---

## 3. アルゴリズム全体像

### 3.1 フローチャート

```
入力: X_raw (生スペクトル), Y (濃度)
    ↓
[ステップ1] 前処理
    ↓
X_pre (前処理済みスペクトル)
    ↓
[ステップ2] LASSO波長選択
    ↓
X_org (選択された波長のみ)
    ↓
[ステップ3] PCAによる再構築
    ↓
X (再構築されたスペクトル)
    ↓
[ステップ4] ランク消去法（R計算）
    ↓
R (干渉成分行列)
    ↓
[ステップ5] NAS抽出
    ↓
Q (Net Analyte Signal行列)
    ↓
[ステップ6] PLSRモデル構築
    ↓
出力: 予測モデル
```

### 3.2 データフロー

```
X_raw (N×H_raw)
    ↓ 前処理
X_pre (N×H_raw)
    ↓ LASSO選択
X_org (N×H_selected)
    ↓ PCA再構築
X (N×H_selected)
    ↓ ランク消去
R (N×H_selected)
    ↓ NAS抽出
Q (N×H_selected)
```

---

## 4. ステップバイステップ詳細解説

### ステップ1: データ前処理

#### 目的
生スペクトルデータからノイズとベースライン変動を除去

#### 入力
- **X_raw**: 生スペクトル行列 (N × H_raw)
  - N: サンプル数（例: 100）
  - H_raw: 全波長点数（例: 1000）
- **Y**: 濃度行列 (N × M)
  - M: 目的成分の数（通常は1）

#### 処理内容

**1.1 Savitzky-Golay平滑化**

```python
from scipy.signal import savgol_filter

# パラメータ
window_length = 11  # 窓サイズ（奇数）
polyorder = 2       # 多項式次数

# 各サンプルに適用
X_smooth = np.zeros_like(X_raw)
for i in range(N):
    X_smooth[i] = savgol_filter(X_raw[i], window_length, polyorder)
```

**数学的背景**:
局所多項式フィッティングによるノイズ除去

**1.2 標準正規変換（SNV: Standard Normal Variate）**

```python
def snv_transform(X):
    """
    SNV変換: 各スペクトルを平均0、標準偏差1に正規化
    """
    X_snv = np.zeros_like(X)
    for i in range(X.shape[0]):
        mean = X[i].mean()
        std = X[i].std()
        X_snv[i] = (X[i] - mean) / std
    return X_snv

X_pre = snv_transform(X_smooth)
```

**数学的表現**:
```
x_i,pre = (x_i - mean(x_i)) / std(x_i)
```

**1.3 その他の前処理オプション**

- **多重散乱補正（MSC）**: ベースライン補正
- **微分**: 1次、2次微分（ベースライン除去）
- **デトレンディング**: 線形トレンド除去

#### 出力
- **X_pre**: 前処理済みスペクトル (N × H_raw)

---

### ステップ2: LASSO波長選択

#### 目的
モデルに真に重要な波長のみを選択し、冗長な変数を除去

#### LASSO回帰の原理

**2.1 最適化問題**

LASSO（Least Absolute Shrinkage and Selection Operator）は以下を最小化：

```
β = argmin_β [ Σ(y_i - Σβ_j x_ij)² + λ Σ|β_j| ]
           = argmin_β [ ||y - Xβ||² + λ||β||₁ ]
```

ここで：
- **第1項**: 残差平方和（予測誤差）
- **第2項**: L1正則化項（係数の絶対値の和）
- **λ**: 正則化パラメータ（収縮の強さ）

**2.2 L1正則化の効果**

λを大きくすると：
1. より多くの係数がゼロになる
2. 変数選択が行われる
3. モデルが簡素化される

**2.3 実装手順**

```python
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# ステップ2.1: データのスケーリング
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pre)

# ステップ2.2: LASSO with Cross-Validation
lasso = LassoCV(
    cv=10,              # 10分割交差検証
    max_iter=10000,     # 最大反復回数
    tol=1e-4,           # 収束判定
    n_jobs=-1           # 並列処理
)

# ステップ2.3: 各成分に対してLASSO実行
y = Y[:, 0]  # 最初の成分（通常は1成分のみ）
lasso.fit(X_scaled, y)

# ステップ2.4: 最適λの取得
lambda_opt = lasso.alpha_
print(f"最適λ: {lambda_opt}")

# ステップ2.5: 係数の取得
coefficients = lasso.coef_

# ステップ2.6: ゼロでない係数のインデックス
selected_indices = np.where(coefficients != 0)[0]
print(f"選択された波長数: {len(selected_indices)} / {H_raw}")

# ステップ2.7: 選択された波長のみ抽出
X_org = X_pre[:, selected_indices]
```

**2.4 λの選択方法**

交差検証によるλの選択：

```python
# 複数のλを試す
alphas = np.logspace(-4, 1, 100)

# 各λでのMSEを計算
mse_path = []
for alpha in alphas:
    lasso_temp = Lasso(alpha=alpha)
    scores = cross_val_score(lasso_temp, X_scaled, y, 
                            cv=10, scoring='neg_mean_squared_error')
    mse_path.append(-scores.mean())

# 最小MSEを与えるλを選択
optimal_alpha = alphas[np.argmin(mse_path)]
```

**2.5 可視化**

```python
import matplotlib.pyplot as plt

# 係数のパス
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左: 係数の値
ax = axes[0]
ax.plot(wavelengths, coefficients, linewidth=2)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('波長 (nm)')
ax.set_ylabel('LASSO係数')
ax.set_title('LASSO回帰係数')
ax.grid(True, alpha=0.3)

# 右: 選択された波長
ax = axes[1]
ax.plot(wavelengths, X_pre[0], alpha=0.3, label='元のスペクトル')
for idx in selected_indices:
    ax.axvline(x=wavelengths[idx], color='red', alpha=0.5)
ax.set_xlabel('波長 (nm)')
ax.set_ylabel('吸光度')
ax.set_title(f'選択された波長 (n={len(selected_indices)})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

#### 出力
- **X_org**: 選択された波長のみのスペクトル (N × H_selected)
- **selected_indices**: 選択された波長のインデックス
- **lambda_opt**: 最適正則化パラメータ

---

### ステップ3: PCAによる再構築

#### 目的
X_orgのランクが完全でない場合に、データを再構築して計算の安定性を確保

#### 3.1 なぜPCA再構築が必要か？

1. **ランク不足**: サンプル数 < 波長数 の場合
2. **数値安定性**: 後のステップでの逆行列計算
3. **ノイズ除去**: 主要な主成分のみを保持

#### 3.2 PCA再構築の手順

**ステップ3.1: データの正規化**

```python
# 平均と標準偏差の計算
X_mean = X_org.mean(axis=0)  # 各波長の平均
X_std = X_org.std(axis=0)    # 各波長の標準偏差

# 正規化
U = (X_org - X_mean) / X_std
```

**ステップ3.2: 共分散行列の計算**

```python
# 共分散行列 (H_selected × H_selected)
Sigma = (1 / (N - 1)) * (U.T @ U)
```

**ステップ3.3: 固有値分解**

```python
# 固有値と固有ベクトル
eigenvalues, eigenvectors = np.linalg.eigh(Sigma)

# 降順にソート
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
```

**ステップ3.4: 主成分数の決定**

```python
# 累積寄与率
cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)

# 95%の分散を説明する主成分数
n_components = np.where(cumulative_variance >= 0.95)[0][0] + 1
print(f"使用する主成分数: {n_components}")
```

**ステップ3.5: 投影とスコア計算**

```python
# 主成分ベクトル (H_selected × n_components)
P = eigenvectors[:, :n_components]

# スコア行列 (N × n_components)
R_scores = U @ P
```

**ステップ3.6: データの再構築**

```python
# 再構築 (N × H_selected)
X_reconstructed = R_scores @ P.T

# 元のスケールに戻す
X = X_reconstructed * X_std + X_mean
```

**3.3 完全な実装**

```python
def pca_reconstruction(X_org, variance_threshold=0.95):
    """
    PCAによるデータ再構築
    
    Parameters
    ----------
    X_org : array (N, H_selected)
        元のデータ
    variance_threshold : float
        保持する分散の割合
    
    Returns
    -------
    X_reconstructed : array (N, H_selected)
        再構築されたデータ
    n_components : int
        使用した主成分数
    """
    N, H = X_org.shape
    
    # 正規化
    X_mean = X_org.mean(axis=0)
    X_std = X_org.std(axis=0)
    U = (X_org - X_mean) / X_std
    
    # 共分散行列
    Sigma = (1 / (N - 1)) * (U.T @ U)
    
    # 固有値分解
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 主成分数の決定
    cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    n_components = np.where(cumulative_variance >= variance_threshold)[0][0] + 1
    
    # 投影
    P = eigenvectors[:, :n_components]
    R_scores = U @ P
    
    # 再構築
    X_reconstructed = R_scores @ P.T
    X_reconstructed = X_reconstructed * X_std + X_mean
    
    return X_reconstructed, n_components

# 使用例
X, n_components = pca_reconstruction(X_org, variance_threshold=0.95)
print(f"再構築完了: {n_components}個の主成分を使用")
```

#### 出力
- **X**: 再構築されたスペクトル (N × H_selected)
- **n_components**: 使用した主成分数

---

### ステップ4: ランク消去法による干渉成分の特定

#### 目的
スペクトルを目的成分（Q）と干渉成分（R）に分離

#### 4.1 理論的背景

スペクトル行列Xは以下のように分解できます：

```
X = Q + R
```

ここで：
- **Q**: 目的成分（COI）に関連するシグナル
- **R**: 干渉成分（OICs）に関連するシグナル

**ランク消去法の原理**:
Yの情報を使って、Xから「Yと相関しない部分」を抽出する

#### 4.2 詳細な計算手順

**ステップ4.1: Xの擬似逆行列を計算**

```python
# Moore-Penrose擬似逆行列
X_pinv = np.linalg.pinv(X)
print(f"X_pinv形状: {X_pinv.shape}")  # (H_selected, N)
```

**ステップ4.2: Yの投影を計算**

```python
# YをXの空間に投影
# y_tilde: Xによって張られる部分空間へのyの投影
y = Y[:, 0]  # 最初の成分（N,）
y_tilde = X @ X_pinv @ y  # (N,)

print(f"y_tilde形状: {y_tilde.shape}")
```

**数学的意味**:
- `y_tilde` は、Xの列空間内でyに最も近い点
- `y - y_tilde` は、Xと直交する残差

**ステップ4.3: スケーリングベクトルdの定義**

```python
# dは、Xの行の線形結合を表すベクトル
# 通常は、すべての要素が1のベクトル（平均スペクトル）
d = np.ones(N)

print(f"d形状: {d.shape}")  # (N,)
```

**ステップ4.4: スケーリング係数αの計算**

```python
# α = 1 / (d^T X^- y_tilde)
denominator = d.T @ X_pinv @ y_tilde
alpha = 1.0 / denominator

print(f"α = {alpha:.6f}")
```

**数学的意味**:
- αは、y_tildeをdの方向に正規化するスケール因子

**ステップ4.5: 干渉成分行列Rの計算**

```python
# R = X - α * y_tilde * d^T
# y_tildeを列ベクトル、dを行ベクトルとして外積を計算
R = X - alpha * np.outer(y_tilde, d)

print(f"R形状: {R.shape}")  # (N, H_selected)
```

**数学的意味**:
- `α * y_tilde * d^T` は、目的成分Qの推定値
- `R = X - Q` は、残りの干渉成分

**4.3 完全な実装**

```python
def calculate_interference_matrix(X, Y):
    """
    ランク消去法により干渉成分行列Rを計算
    
    Parameters
    ----------
    X : array (N, H_selected)
        再構築されたスペクトル行列
    Y : array (N, M)
        濃度行列（通常M=1）
    
    Returns
    -------
    R : array (N, H_selected)
        干渉成分行列
    alpha : float
        スケーリング係数
    y_tilde : array (N,)
        投影された濃度
    """
    N, H = X.shape
    
    # ステップ1: 擬似逆行列
    X_pinv = np.linalg.pinv(X)
    
    # ステップ2: 投影
    y = Y[:, 0] if Y.ndim > 1 else Y
    y_tilde = X @ X_pinv @ y
    
    # ステップ3: スケーリングベクトル
    d = np.ones(N)
    
    # ステップ4: スケーリング係数
    denominator = d.T @ X_pinv @ y_tilde
    if np.abs(denominator) < 1e-10:
        print("Warning: denominator too small, using regularization")
        denominator = 1e-10 if denominator >= 0 else -1e-10
    alpha = 1.0 / denominator
    
    # ステップ5: 干渉成分
    R = X - alpha * np.outer(y_tilde, d)
    
    return R, alpha, y_tilde

# 使用例
R, alpha, y_tilde = calculate_interference_matrix(X, Y)
print(f"干渉成分行列R: {R.shape}")
print(f"スケーリング係数α: {alpha:.6f}")
```

**4.4 検証**

```python
# Rのランクを確認
rank_R = np.linalg.matrix_rank(R)
rank_X = np.linalg.matrix_rank(X)

print(f"rank(X) = {rank_X}")
print(f"rank(R) = {rank_R}")
print(f"rank(R) < rank(X): {rank_R < rank_X}")  # Trueであるべき
```

#### 出力
- **R**: 干渉成分行列 (N × H_selected)
- **alpha**: スケーリング係数
- **y_tilde**: 投影された濃度ベクトル (N,)

---

### ステップ5: NAS（Net Analyte Signal）の抽出

#### 目的
各サンプルのスペクトルから、干渉成分を除去してCOIの純粋なシグナルを抽出

#### 5.1 NAS計算の原理

NASベクトル **q_i** は、元のスペクトル **x_i** から干渉成分の影響を投影除去したものです：

```
q_i = (I - R^T(RR^T)^-1 R) x_i
```

または簡略化して：

```
q_i = (I - R^T(R^T)^-1) x_i
```

ここで：
- **I**: 単位行列 (H_selected × H_selected)
- **R^T**: Rの転置 (H_selected × N)
- **(R^T)^-1**: R^Tの擬似逆行列
- **x_i**: i番目のサンプルのスペクトル (H_selected,)

#### 5.2 詳細な計算手順

**ステップ5.1: R^Tの擬似逆行列を計算**

```python
# Rの転置
R_T = R.T  # (H_selected, N)

# 擬似逆行列
R_T_pinv = np.linalg.pinv(R_T)  # (N, H_selected)

print(f"R^T形状: {R_T.shape}")
print(f"(R^T)^-形状: {R_T_pinv.shape}")
```

**ステップ5.2: 投影行列の計算**

```python
# 投影行列 P = R^T (R^T)^-
P = R_T @ R_T_pinv  # (H_selected, H_selected)

print(f"投影行列P形状: {P.shape}")

# Pは冪等行列であるべき（P @ P ≈ P）
P_squared = P @ P
print(f"冪等性チェック: ||P² - P|| = {np.linalg.norm(P_squared - P):.6e}")
```

**ステップ5.3: 直交投影行列の計算**

```python
# 単位行列
I = np.eye(H_selected)

# 直交投影行列 P_orth = I - P
P_orth = I - P  # (H_selected, H_selected)

print(f"直交投影行列形状: {P_orth.shape}")
```

**数学的意味**:
- **P**: Rの列空間への投影
- **P_orth**: Rの列空間に直交する部分空間への投影
- **P_orth @ x_i**: x_iからRの影響を除去

**ステップ5.4: 各サンプルのNASを計算**

```python
# Q行列の初期化
Q = np.zeros_like(X)  # (N, H_selected)

# 各サンプルに対してNAS計算
for i in range(N):
    x_i = X[i, :]  # i番目のスペクトル (H_selected,)
    q_i = P_orth @ x_i  # NASベクトル (H_selected,)
    Q[i, :] = q_i

print(f"Q行列形状: {Q.shape}")
```

**5.5 ベクトル化実装（高速化）**

```python
# ループを使わないベクトル化実装
Q = (P_orth @ X.T).T  # (N, H_selected)

# または
Q = X @ P_orth.T  # (N, H_selected)
```

**5.6 完全な実装**

```python
def extract_nas(X, R):
    """
    Net Analyte Signal (NAS) の抽出
    
    Parameters
    ----------
    X : array (N, H_selected)
        スペクトル行列
    R : array (N, H_selected)
        干渉成分行列
    
    Returns
    -------
    Q : array (N, H_selected)
        NAS行列
    P_orth : array (H_selected, H_selected)
        直交投影行列
    """
    N, H = X.shape
    
    # ステップ1: R^Tの擬似逆行列
    R_T = R.T
    R_T_pinv = np.linalg.pinv(R_T)
    
    # ステップ2: 投影行列
    P = R_T @ R_T_pinv
    
    # ステップ3: 直交投影行列
    I = np.eye(H)
    P_orth = I - P
    
    # ステップ4: NAS計算（ベクトル化）
    Q = X @ P_orth.T
    
    # 検証: QとRの直交性
    orthogonality = np.linalg.norm(Q @ R.T)
    print(f"直交性チェック: ||Q R^T|| = {orthogonality:.6e}")
    
    return Q, P_orth

# 使用例
Q, P_orth = extract_nas(X, R)
print(f"NAS行列Q: {Q.shape}")
```

**5.7 NASの検証**

```python
# 検証1: X = Q + R の確認
X_reconstructed = Q + R
reconstruction_error = np.linalg.norm(X - X_reconstructed)
print(f"再構築誤差: {reconstruction_error:.6e}")

# 検証2: QとRの直交性
orthogonality = np.abs(np.trace(Q.T @ R)) / (np.linalg.norm(Q) * np.linalg.norm(R))
print(f"直交性指標: {orthogonality:.6e}")  # 0に近いほど良い

# 検証3: Qのエネルギー
Q_energy = np.linalg.norm(Q, 'fro') ** 2
X_energy = np.linalg.norm(X, 'fro') ** 2
energy_ratio = Q_energy / X_energy
print(f"Q/Xエネルギー比: {energy_ratio:.4f}")
```

**5.8 可視化**

```python
import matplotlib.pyplot as plt

# サンプル選択
sample_idx = 0

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 元のスペクトル
ax = axes[0, 0]
ax.plot(X[sample_idx], label='元のスペクトル X', linewidth=2)
ax.set_xlabel('波長インデックス')
ax.set_ylabel('吸光度')
ax.set_title(f'サンプル {sample_idx}: 元のスペクトル')
ax.legend()
ax.grid(True, alpha=0.3)

# 干渉成分
ax = axes[0, 1]
ax.plot(R[sample_idx], label='干渉成分 R', color='orange', linewidth=2)
ax.set_xlabel('波長インデックス')
ax.set_ylabel('吸光度')
ax.set_title(f'サンプル {sample_idx}: 干渉成分')
ax.legend()
ax.grid(True, alpha=0.3)

# NAS
ax = axes[1, 0]
ax.plot(Q[sample_idx], label='NAS Q', color='green', linewidth=2)
ax.set_xlabel('波長インデックス')
ax.set_ylabel('吸光度')
ax.set_title(f'サンプル {sample_idx}: Net Analyte Signal')
ax.legend()
ax.grid(True, alpha=0.3)

# 比較
ax = axes[1, 1]
ax.plot(X[sample_idx], label='X', alpha=0.7, linewidth=2)
ax.plot(R[sample_idx], label='R', alpha=0.7, linewidth=2)
ax.plot(Q[sample_idx], label='Q', alpha=0.7, linewidth=2)
ax.set_xlabel('波長インデックス')
ax.set_ylabel('吸光度')
ax.set_title(f'サンプル {sample_idx}: 比較')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

#### 出力
- **Q**: NAS行列 (N × H_selected)
- **P_orth**: 直交投影行列 (H_selected × H_selected)

---

### ステップ6: PLSRモデルの構築

#### 目的
抽出されたNAS（Q）を使用して、目的成分の定量モデルを構築

#### 6.1 なぜPLSRを使うのか？

1. **多重共線性に強い**: NAS後も波長間の相関が残る
2. **次元削減**: 潜在変数を使用
3. **予測精度**: 高い予測性能
4. **解釈性**: VIP（Variable Importance in Projection）

#### 6.2 PLSR実装

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error

# ステップ6.1: 最適な潜在変数数の決定
max_components = min(20, Q.shape[1], Q.shape[0] - 1)
mse_cv = []

for n_comp in range(1, max_components + 1):
    pls = PLSRegression(n_components=n_comp)
    y_cv = cross_val_predict(pls, Q, y, cv=10)
    mse = mean_squared_error(y, y_cv)
    mse_cv.append(mse)

# 最適成分数
optimal_n_comp = np.argmin(mse_cv) + 1
print(f"最適潜在変数数: {optimal_n_comp}")

# ステップ6.2: 最終モデルの構築
pls_model = PLSRegression(n_components=optimal_n_comp)
pls_model.fit(Q, y)

# ステップ6.3: 予測
y_pred = pls_model.predict(Q)

# ステップ6.4: 評価
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"R² = {r2:.6f}")
print(f"RMSE = {rmse:.6f}")
```

**6.3 VIP（Variable Importance in Projection）の計算**

```python
def calculate_vip(pls_model):
    """
    VIPスコアの計算
    """
    t = pls_model.x_scores_  # Xスコア
    w = pls_model.x_weights_  # X重み
    q = pls_model.y_loadings_  # Yローディング
    
    p, h = w.shape
    vips = np.zeros((p,))
    
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    
    for i in range(p):
        weight = np.array([(w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
    
    return vips

vip_scores = calculate_vip(pls_model)
print(f"VIPスコア: min={vip_scores.min():.3f}, max={vip_scores.max():.3f}")
```

#### 出力
- **pls_model**: 訓練済みPLSRモデル
- **y_pred**: 予測濃度 (N,)
- **R²**: 決定係数
- **RMSE**: 二乗平均平方根誤差
- **vip_scores**: VIPスコア (H_selected,)

---

## 5. 数学的定式化

### 5.1 完全な数式の流れ

#### 入力

```
X_raw ∈ ℝ^(N×H_raw)  : 生スペクトル
Y ∈ ℝ^(N×M)          : 濃度行列
```

#### ステップ1: 前処理

```
X_pre = Preprocess(X_raw)
```

#### ステップ2: LASSO

```
β* = argmin_β [ ||y - X_pre β||² + λ||β||₁ ]
S = {j : β_j* ≠ 0}
X_org = X_pre[:, S]
```

#### ステップ3: PCA再構築

```
U = (X_org - μ) / σ
Σ = (1/(N-1)) U^T U
Σ = PΛP^T  (固有値分解)
X = (UP_k P_k^T) σ + μ
```

#### ステップ4: ランク消去

```
X^- = (X^T X)^(-1) X^T  (擬似逆行列)
ỹ = XX^- y
α = 1 / (d^T X^- ỹ)
R = X - α ỹd^T
```

#### ステップ5: NAS抽出

```
P_orth = I - R^T(R^T)^(-1)
Q = XP_orth^T
```

#### ステップ6: PLSR

```
y_pred = f_PLSR(Q)
```

### 5.2 行列の次元

| 行列 | 次元 | 説明 |
|------|------|------|
| X_raw | N × H_raw | 生スペクトル |
| X_pre | N × H_raw | 前処理済み |
| X_org | N × H_sel | LASSO選択後 |
| X | N × H_sel | PCA再構築後 |
| R | N × H_sel | 干渉成分 |
| Q | N × H_sel | NAS |
| Y | N × M | 濃度 |

---

## 6. Python完全実装

### 6.1 統合クラス実装

```python
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')


class LassoNAS:
    """
    LASSO-NAS特徴抽出の完全実装
    
    Parameters
    ----------
    preprocess_method : str, default='snv'
        前処理方法 ('snv', 'msc', 'savgol', 'none')
    
    lasso_cv : int, default=10
        LASSO交差検証の分割数
    
    pca_variance : float, default=0.95
        PCA再構築で保持する分散の割合
    
    plsr_max_components : int, default=20
        PLSRの最大成分数
    
    Attributes
    ----------
    selected_wavelengths_ : array
        選択された波長のインデックス
    
    Q_ : array
        抽出されたNAS行列
    
    R_ : array
        干渉成分行列
    
    pls_model_ : PLSRegression
        訓練済みPLSRモデル
    """
    
    def __init__(
        self,
        preprocess_method='snv',
        lasso_cv=10,
        pca_variance=0.95,
        plsr_max_components=20
    ):
        self.preprocess_method = preprocess_method
        self.lasso_cv = lasso_cv
        self.pca_variance = pca_variance
        self.plsr_max_components = plsr_max_components
        
        # 内部状態
        self.selected_wavelengths_ = None
        self.X_mean_ = None
        self.X_std_ = None
        self.lasso_model_ = None
        self.Q_ = None
        self.R_ = None
        self.P_orth_ = None
        self.pls_model_ = None
        self.n_components_pca_ = None
        self.n_components_pls_ = None
    
    def _preprocess(self, X):
        """ステップ1: 前処理"""
        if self.preprocess_method == 'snv':
            return self._snv(X)
        elif self.preprocess_method == 'savgol':
            return self._savgol(X)
        elif self.preprocess_method == 'msc':
            return self._msc(X)
        else:
            return X.copy()
    
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
    
    def _msc(self, X):
        """多重散乱補正"""
        ref = np.mean(X, axis=0)
        X_msc = np.zeros_like(X)
        for i in range(X.shape[0]):
            fit = np.polyfit(ref, X[i], 1)
            X_msc[i] = (X[i] - fit[1]) / fit[0]
        return X_msc
    
    def _lasso_selection(self, X, y):
        """ステップ2: LASSO波長選択"""
        # スケーリング
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # LASSO with CV
        lasso = LassoCV(cv=self.lasso_cv, max_iter=10000, n_jobs=-1)
        lasso.fit(X_scaled, y)
        
        # 選択された波長
        coef = lasso.coef_
        selected = np.where(coef != 0)[0]
        
        if len(selected) == 0:
            print("Warning: No wavelengths selected, using all")
            selected = np.arange(X.shape[1])
        
        self.lasso_model_ = lasso
        self.selected_wavelengths_ = selected
        
        return X[:, selected]
    
    def _pca_reconstruction(self, X):
        """ステップ3: PCA再構築"""
        N, H = X.shape
        
        # 正規化
        self.X_mean_ = X.mean(axis=0)
        self.X_std_ = X.std(axis=0)
        U = (X - self.X_mean_) / self.X_std_
        
        # 共分散行列と固有値分解
        Sigma = (1 / (N - 1)) * (U.T @ U)
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
        
        # 降順ソート
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 主成分数の決定
        cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        n_comp = np.where(cumvar >= self.pca_variance)[0][0] + 1
        self.n_components_pca_ = n_comp
        
        # 再構築
        P = eigenvectors[:, :n_comp]
        R_scores = U @ P
        X_recon = R_scores @ P.T
        X_recon = X_recon * self.X_std_ + self.X_mean_
        
        return X_recon
    
    def _calculate_interference(self, X, y):
        """ステップ4: ランク消去法"""
        N = X.shape[0]
        
        # 擬似逆行列
        X_pinv = np.linalg.pinv(X)
        
        # 投影
        y_tilde = X @ X_pinv @ y
        
        # スケーリング
        d = np.ones(N)
        denom = d.T @ X_pinv @ y_tilde
        if np.abs(denom) < 1e-10:
            denom = 1e-10 if denom >= 0 else -1e-10
        alpha = 1.0 / denom
        
        # 干渉成分
        R = X - alpha * np.outer(y_tilde, d)
        
        return R
    
    def _extract_nas(self, X, R):
        """ステップ5: NAS抽出"""
        H = X.shape[1]
        
        # 擬似逆行列
        R_T = R.T
        R_T_pinv = np.linalg.pinv(R_T)
        
        # 投影行列
        P = R_T @ R_T_pinv
        I = np.eye(H)
        P_orth = I - P
        
        # NAS計算
        Q = X @ P_orth.T
        
        self.P_orth_ = P_orth
        
        return Q
    
    def _build_plsr(self, Q, y):
        """ステップ6: PLSRモデル構築"""
        # 最適成分数の決定
        max_comp = min(self.plsr_max_components, Q.shape[1], Q.shape[0] - 1)
        mse_cv = []
        
        for n_comp in range(1, max_comp + 1):
            pls = PLSRegression(n_components=n_comp)
            scores = cross_val_score(
                pls, Q, y, cv=10,
                scoring='neg_mean_squared_error'
            )
            mse_cv.append(-scores.mean())
        
        # 最適成分数
        self.n_components_pls_ = np.argmin(mse_cv) + 1
        
        # 最終モデル
        pls = PLSRegression(n_components=self.n_components_pls_)
        pls.fit(Q, y)
        
        self.pls_model_ = pls
        
        return pls
    
    def fit(self, X_raw, Y):
        """
        モデルの訓練
        
        Parameters
        ----------
        X_raw : array (N, H_raw)
            生スペクトルデータ
        Y : array (N, M) or (N,)
            濃度データ
        
        Returns
        -------
        self : LassoNAS
        """
        # 濃度の形状確認
        y = Y[:, 0] if Y.ndim > 1 else Y
        
        print("=" * 60)
        print("LASSO-NAS 特徴抽出開始")
        print("=" * 60)
        
        # ステップ1: 前処理
        print("\n[ステップ1] 前処理...")
        X_pre = self._preprocess(X_raw)
        print(f"  前処理完了: {X_pre.shape}")
        
        # ステップ2: LASSO選択
        print("\n[ステップ2] LASSO波長選択...")
        X_org = self._lasso_selection(X_pre, y)
        print(f"  選択された波長数: {len(self.selected_wavelengths_)} / {X_raw.shape[1]}")
        print(f"  最適λ: {self.lasso_model_.alpha_:.6f}")
        
        # ステップ3: PCA再構築
        print("\n[ステップ3] PCA再構築...")
        X = self._pca_reconstruction(X_org)
        print(f"  使用した主成分数: {self.n_components_pca_}")
        
        # ステップ4: 干渉成分計算
        print("\n[ステップ4] ランク消去法...")
        self.R_ = self._calculate_interference(X, y)
        rank_R = np.linalg.matrix_rank(self.R_)
        print(f"  干渉成分行列R: rank={rank_R}")
        
        # ステップ5: NAS抽出
        print("\n[ステップ5] NAS抽出...")
        self.Q_ = self._extract_nas(X, self.R_)
        
        # 検証
        recon_error = np.linalg.norm(X - (self.Q_ + self.R_))
        print(f"  再構築誤差: {recon_error:.6e}")
        
        orthogonality = np.linalg.norm(self.Q_ @ self.R_.T)
        print(f"  直交性: {orthogonality:.6e}")
        
        # ステップ6: PLSRモデル
        print("\n[ステップ6] PLSRモデル構築...")
        self._build_plsr(self.Q_, y)
        print(f"  使用した潜在変数数: {self.n_components_pls_}")
        
        # 訓練性能
        y_pred = self.pls_model_.predict(self.Q_).ravel()
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        print("\n" + "=" * 60)
        print("訓練完了")
        print("=" * 60)
        print(f"R² = {r2:.6f}")
        print(f"RMSE = {rmse:.6f}")
        print("=" * 60)
        
        return self
    
    def predict(self, X_raw):
        """
        新しいスペクトルの予測
        
        Parameters
        ----------
        X_raw : array (N_new, H_raw)
            新しいスペクトルデータ
        
        Returns
        -------
        y_pred : array (N_new,)
            予測濃度
        """
        # ステップ1: 前処理
        X_pre = self._preprocess(X_raw)
        
        # ステップ2: 波長選択
        X_org = X_pre[:, self.selected_wavelengths_]
        
        # ステップ3: PCA変換（訓練時のパラメータを使用）
        U = (X_org - self.X_mean_) / self.X_std_
        # 注: 完全な再構築は省略可能
        X = X_org  # 簡略化
        
        # ステップ5: NAS抽出
        Q = X @ self.P_orth_.T
        
        # ステップ6: PLSR予測
        y_pred = self.pls_model_.predict(Q).ravel()
        
        return y_pred
    
    def get_feature_importance(self):
        """
        特徴量重要度の取得
        
        Returns
        -------
        importance : dict
            各種重要度指標
        """
        # LASSO係数
        lasso_coef = np.zeros(len(self.selected_wavelengths_))
        lasso_coef = self.lasso_model_.coef_[self.selected_wavelengths_]
        
        # VIP計算
        vip = self._calculate_vip()
        
        return {
            'lasso_coefficients': lasso_coef,
            'vip_scores': vip,
            'selected_wavelengths': self.selected_wavelengths_
        }
    
    def _calculate_vip(self):
        """VIPスコアの計算"""
        pls = self.pls_model_
        t = pls.x_scores_
        w = pls.x_weights_
        q = pls.y_loadings_
        
        p, h = w.shape
        vips = np.zeros((p,))
        
        s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)
        
        for i in range(p):
            weight = np.array([
                (w[i,j] / np.linalg.norm(w[:,j]))**2 
                for j in range(h)
            ])
            vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
        
        return vips


# 使用例
if __name__ == "__main__":
    # シミュレーションデータ
    np.random.seed(42)
    
    N = 100  # サンプル数
    H = 200  # 波長数
    
    # 生スペクトル生成
    X_raw = np.random.randn(N, H)
    
    # 濃度生成
    Y = np.random.uniform(0, 10, N)
    
    # モデル作成
    model = LassoNAS(
        preprocess_method='snv',
        lasso_cv=10,
        pca_variance=0.95,
        plsr_max_components=20
    )
    
    # 訓練
    model.fit(X_raw, Y)
    
    # 予測
    y_pred = model.predict(X_raw)
    
    # 評価
    r2 = r2_score(Y, y_pred)
    print(f"\n最終 R² = {r2:.6f}")
```

---

## 7. 実例による実演

### 7.1 シミュレーションデータでの完全な例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# データ生成
np.random.seed(42)

N = 150  # サンプル数
H = 300  # 波長数

# 波長軸
wavelengths = np.linspace(400, 2500, H)

# 目的成分の純粋スペクトル（ガウシアンピーク）
coi_spectrum = (
    np.exp(-((wavelengths - 1000) ** 2) / (2 * 100 ** 2)) +
    0.5 * np.exp(-((wavelengths - 1500) ** 2) / (2 * 80 ** 2))
)

# 干渉成分の純粋スペクトル
oic_spectrum1 = np.exp(-((wavelengths - 800) ** 2) / (2 * 150 ** 2))
oic_spectrum2 = np.exp(-((wavelengths - 1800) ** 2) / (2 * 120 ** 2))

# 濃度生成
c_coi = np.random.uniform(1, 10, N)
c_oic1 = np.random.uniform(0.5, 3, N)
c_oic2 = np.random.uniform(0.3, 2, N)

# スペクトル生成
X_raw = (
    c_coi[:, None] * coi_spectrum +
    c_oic1[:, None] * oic_spectrum1 +
    c_oic2[:, None] * oic_spectrum2 +
    np.random.normal(0, 0.02, (N, H))
)

Y = c_coi

# 訓練/テスト分割
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, Y, test_size=0.3, random_state=42
)

print("データ形状:")
print(f"  訓練: X={X_train.shape}, y={y_train.shape}")
print(f"  テスト: X={X_test.shape}, y={y_test.shape}")

# モデル訓練
model = LassoNAS(
    preprocess_method='snv',
    lasso_cv=10,
    pca_variance=0.95,
    plsr_max_components=15
)

model.fit(X_train, y_train)

# テスト予測
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 評価
r2_train = r2_score(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\n" + "=" * 60)
print("最終評価")
print("=" * 60)
print(f"訓練: R²={r2_train:.6f}, RMSE={rmse_train:.6f}")
print(f"テスト: R²={r2_test:.6f}, RMSE={rmse_test:.6f}")
print("=" * 60)

# 可視化
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. 元のスペクトル
ax = axes[0, 0]
for i in range(min(10, len(X_train))):
    ax.plot(wavelengths, X_train[i], alpha=0.5)
ax.set_xlabel('波長 (nm)')
ax.set_ylabel('吸光度')
ax.set_title('元のスペクトル（訓練セット）')
ax.grid(True, alpha=0.3)

# 2. 選択された波長
ax = axes[0, 1]
selected_mask = np.zeros(H, dtype=bool)
selected_mask[model.selected_wavelengths_] = True
ax.plot(wavelengths, X_train[0], alpha=0.3, label='スペクトル')
for wl in wavelengths[selected_mask]:
    ax.axvline(x=wl, color='red', alpha=0.3)
ax.set_xlabel('波長 (nm)')
ax.set_ylabel('吸光度')
ax.set_title(f'LASSO選択波長 (n={len(model.selected_wavelengths_)})')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. NAS vs 元のスペクトル
ax = axes[0, 2]
sample_idx = 0
ax.plot(model.Q_[sample_idx], label='NAS', linewidth=2)
ax.plot(X_train[sample_idx, model.selected_wavelengths_], 
        label='元のスペクトル', alpha=0.6)
ax.set_xlabel('波長インデックス')
ax.set_ylabel('吸光度')
ax.set_title(f'サンプル {sample_idx}: NAS vs 元')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. 訓練セット予測
ax = axes[1, 0]
ax.scatter(y_train, y_pred_train, alpha=0.6, s=50)
ax.plot([y_train.min(), y_train.max()], 
        [y_train.min(), y_train.max()], 
        'r--', linewidth=2)
ax.set_xlabel('真の濃度')
ax.set_ylabel('予測濃度')
ax.set_title(f'訓練セット (R²={r2_train:.4f})')
ax.grid(True, alpha=0.3)

# 5. テストセット予測
ax = axes[1, 1]
ax.scatter(y_test, y_pred_test, alpha=0.6, s=50, color='green')
ax.plot([y_test.min(), y_test.max()], 
        [y_test.min(), y_test.max()], 
        'r--', linewidth=2)
ax.set_xlabel('真の濃度')
ax.set_ylabel('予測濃度')
ax.set_title(f'テストセット (R²={r2_test:.4f})')
ax.grid(True, alpha=0.3)

# 6. 残差プロット
ax = axes[1, 2]
residuals_test = y_test - y_pred_test
ax.scatter(y_pred_test, residuals_test, alpha=0.6, s=50, color='purple')
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('予測濃度')
ax.set_ylabel('残差')
ax.set_title('残差プロット（テストセット）')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lasso_nas_complete_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 8. 古典的NASとの比較

### 8.1 主な違い

| 特徴 | 古典的NAS (Lorber 1986) | LASSO-NAS |
|------|------------------------|-----------|
| **波長選択** | なし（全波長使用） | LASSO事前選択 |
| **計算式** | `n = (X^T X)^-1 X^T y` | 多段階プロセス |
| **干渉除去** | 直接計算 | ランク消去法 |
| **数値安定性** | 逆行列計算で不安定 | 選択後で安定 |
| **計算量** | O(H³) | O(H³) + LASSO |
| **解釈性** | 中程度 | 高い（重要波長明確） |

### 8.2 アルゴリズムの比較

**古典的NAS**:
```
X, y → n = (X^T X)^-1 X^T y → 予測
```

**LASSO-NAS**:
```
X_raw, Y → 前処理 → LASSO選択 → PCA → ランク消去 → NAS抽出 → PLSR → 予測
```

### 8.3 適用場面

**古典的NASが適している場合**:
- 波長数が少ない（< 100）
- サンプル数が多い（N >> H）
- 干渉成分が少ない
- 計算速度が重要

**LASSO-NASが適している場合**:
- 波長数が多い（> 100）
- 多重共線性が強い
- 干渉成分が複数
- 予測精度が重要

---

## 9. よくある質問

### Q1: なぜLASSOを先に適用するのか？

**A**: 
1. **次元削減**: 数千の波長を数十〜数百に削減
2. **ノイズ除去**: 無関係な波長を除去
3. **計算安定性**: 後のステップでの逆行列計算が安定
4. **解釈性**: 重要な波長が明確

### Q2: PCA再構築は必須か？

**A**: **必須ではありませんが推奨**

- **必要な場合**:
  - サンプル数 < 選択波長数
  - ランク不足の問題
  - 数値的不安定性

- **省略可能な場合**:
  - サンプル数 >> 選択波長数
  - データが既に良好な条件数

### Q3: ランク消去法の直感的理解は？

**A**: 

1. **Xの空間**にyを投影 → `y_tilde`
2. **y_tildeとdの外積**で「目的成分のみの行列」を作成
3. **X - 目的成分 = 干渉成分R**

つまり、「yと相関する部分」を除去して「残り（干渉）」を得る。

### Q4: QとRの直交性はなぜ重要か？

**A**:

- **直交性**: `Q^T R ≈ 0`
- **意味**: QとRが独立（相関なし）
- **重要性**:
  - Qが純粋にCOIのシグナル
  - Rが純粋に干渉
  - 混ざっていない証拠

### Q5: 実データで性能が出ない場合は？

**A**: 以下を確認：

1. **前処理の選択**
   - SNV, MSC, Savgol-Golayを試す
   - 微分も検討

2. **LASSO λの調整**
   - 交差検証の分割数を変更
   - λの範囲を調整

3. **PCA分散閾値**
   - 0.95 → 0.99に増やす
   - または固定成分数を指定

4. **PLSRの成分数**
   - 過学習の可能性
   - 交差検証で再確認

5. **外れ値の除去**
   - Hotellingのt²統計量
   - Q統計量

---

## 10. まとめ

### 10.1 LASSO-NASの利点

✅ **高次元データに対応**: 数千の波長でも処理可能  
✅ **ロバスト**: ノイズと干渉に強い  
✅ **解釈可能**: 重要波長が明確  
✅ **高精度**: 従来法より予測性能が向上  
✅ **汎用性**: 様々なスペクトル分析に適用可能  

### 10.2 実装のポイント

1. **前処理**: データに適した方法を選択
2. **LASSO**: 交差検証で最適λを決定
3. **PCA**: 分散閾値を適切に設定
4. **ランク消去**: 数値安定性に注意
5. **PLSR**: 過学習に注意

### 10.3 応用分野

- **NIR分光法**: 食品、農業、製薬
- **Raman分光法**: 材料科学、バイオメディカル
- **質量分析**: プロテオミクス、メタボロミクス
- **蛍光分光法**: 環境モニタリング

---

## 参考文献

### 原著論文
1. **この論文**: "Feature Extraction From Spectroscopy Using LASSO and Net Analyte Signal"

### 関連文献
2. Lorber, A. (1986). "Error propagation and figures of merit for quantification by solving matrix equations." *Analytical Chemistry*, 58(6), 1167-1172.

3. Tibshirani, R. (1996). "Regression shrinkage and selection via the lasso." *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.

4. Faber, N. M., & Kowalski, B. R. (1997). "Net analyte signal calculation in multivariate calibration." *Analytical Chemistry*, 69(16), 3451-3459.

---

**作成日**: 2026年2月17日  
**バージョン**: 1.0  
**ライセンス**: CC BY 4.0
