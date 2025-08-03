# トラブルシューティングガイド & FAQ

## 目次

1. [一般的な問題と解決方法](#一般的な問題と解決方法)
2. [インストール関連の問題](#インストール関連の問題)
3. [データ関連の問題](#データ関連の問題)
4. [パフォーマンス関連の問題](#パフォーマンス関連の問題)
5. [特徴量分析の問題](#特徴量分析の問題)
6. [GAT 最適化の問題](#gat最適化の問題)
7. [設定関連の問題](#設定関連の問題)
8. [よくある質問 (FAQ)](#よくある質問-faq)
9. [デバッグ手法](#デバッグ手法)
10. [サポートとコミュニティ](#サポートとコミュニティ)

## 一般的な問題と解決方法

### 問題 1: ModuleNotFoundError が発生する

**症状**:

```
ModuleNotFoundError: No module named 'analysis.feature_analysis'
```

**原因**:

- Python パスが正しく設定されていない
- 必要な依存関係がインストールされていない
- 仮想環境が有効化されていない

**解決方法**:

1. **Python パスの確認**:

```bash
# 現在の作業ディレクトリを確認
pwd

# PYTHONPATHに追加
export PYTHONPATH="${PYTHONPATH}:/path/to/kazoo"

# または、__init__.py ファイルが存在することを確認
find . -name "__init__.py"
```

2. **依存関係の再インストール**:

```bash
# 仮想環境の有効化
source venv/bin/activate

# 依存関係のインストール
pip install -r requirements.txt

# または、開発モードでのインストール
pip install -e .
```

3. **モジュール構造の確認**:

```python
import sys
sys.path.append('/path/to/kazoo')

# または、相対インポートを使用
from .analysis.feature_analysis import FeatureImportanceAnalyzer
```

### 問題 2: メモリ不足エラー

**症状**:

```
MemoryError: Unable to allocate array with shape (100000, 5000)
```

**原因**:

- データセットが利用可能なメモリを超えている
- 非効率なデータ処理
- メモリリークの発生

**解決方法**:

1. **バッチ処理の実装**:

```python
def process_large_dataset_in_batches(features, batch_size=1000):
    """大規模データセットをバッチで処理"""
    n_samples = features.shape[0]
    results = []

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_features = features[start_idx:end_idx]

        # バッチ処理
        batch_result = process_feature_batch(batch_features)
        results.append(batch_result)

        # メモリクリーンアップ
        del batch_features
        import gc
        gc.collect()

    return np.concatenate(results, axis=0)
```

2. **メモリ効率的なデータ型の使用**:

```python
# float64の代わりにfloat32を使用
features = features.astype(np.float32)

# 疎行列を使用（適用可能な場合）
from scipy.sparse import csr_matrix
sparse_features = csr_matrix(features)
```

3. **データのチャンク処理**:

```python
import pandas as pd

# chunksize を指定してデータを読み込み
for chunk in pd.read_csv('large_dataset.csv', chunksize=10000):
    process_chunk(chunk)
```

### 問題 3: 計算が非常に遅い

**症状**:

- 特徴量分析が数時間かかる
- パイプライン実行が予想以上に時間がかかる

**原因**:

- 非効率なアルゴリズム実装
- 並列処理が有効化されていない
- 不適切なデータ構造の使用

**解決方法**:

1. **並列処理の有効化**:

```python
# 設定ファイルで並列処理を有効化
pipeline:
  enable_parallel: true
  max_workers: 8

# または、直接指定
from multiprocessing import Pool
with Pool(processes=8) as pool:
    results = pool.map(process_function, data_chunks)
```

2. **ベクトル化の活用**:

```python
# 遅い: ループを使用
similarities = []
for i in range(len(features1)):
    for j in range(len(features2)):
        sim = cosine_similarity([features1[i]], [features2[j]])[0][0]
        similarities.append(sim)

# 高速: ベクトル化
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(features1, features2)
```

3. **キャッシュの活用**:

```python
# 設定でキャッシュを有効化
pipeline:
  enable_cache: true
  cache_expiry_hours: 24
```

## インストール関連の問題

### PyTorch Geometric のインストール問題

**症状**:

```
ERROR: Failed building wheel for torch-geometric
```

**解決方法**:

1. **PyTorch の互換性確認**:

```bash
# PyTorchのバージョン確認
python -c "import torch; print(torch.__version__)"

# CUDA対応確認
python -c "import torch; print(torch.cuda.is_available())"
```

2. **適切なバージョンの PyTorch Geometric をインストール**:

```bash
# CPU版の場合
pip install torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html

# CUDA 11.6の場合
pip install torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
```

3. **依存関係の個別インストール**:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
pip install torch-geometric
```

### scikit-learn バージョン互換性問題

**症状**:

```
AttributeError: module 'sklearn' has no attribute 'some_function'
```

**解決方法**:

1. **scikit-learn のアップグレード**:

```bash
pip install --upgrade scikit-learn
```

2. **互換性コードの追加**:

```python
try:
    from sklearn.model_selection import cross_val_score
except ImportError:
    from sklearn.cross_validation import cross_val_score
```

## データ関連の問題

### 問題 1: 不正なデータ形状エラー

**症状**:

```
ValueError: X has 150 features, but expected 200 features
```

**診断**:

```python
def diagnose_data_shape_mismatch(features, expected_shape):
    """データ形状の不整合を診断"""
    print(f"実際の形状: {features.shape}")
    print(f"期待される形状: {expected_shape}")

    if len(features.shape) != len(expected_shape):
        print("次元数が異なります")

    for i, (actual, expected) in enumerate(zip(features.shape, expected_shape)):
        if actual != expected:
            print(f"次元 {i}: 実際={actual}, 期待={expected}")

    # データ型確認
    print(f"データ型: {features.dtype}")

    # 欠損値確認
    if hasattr(features, 'isna'):
        print(f"欠損値数: {features.isna().sum().sum()}")
    else:
        print(f"欠損値数: {np.isnan(features).sum()}")
```

**解決方法**:

1. **特徴量名と形状の確認**:

```python
def validate_feature_consistency(features, feature_names):
    """特徴量の一貫性を検証"""
    if features.shape[1] != len(feature_names):
        print(f"警告: 特徴量数 ({features.shape[1]}) と特徴量名数 ({len(feature_names)}) が一致しません")

        # 不足している特徴量を特定
        if features.shape[1] < len(feature_names):
            missing_features = feature_names[features.shape[1]:]
            print(f"不足している特徴量: {missing_features}")
        else:
            extra_features = features.shape[1] - len(feature_names)
            print(f"余分な特徴量数: {extra_features}")

    return features.shape[1] == len(feature_names)
```

2. **データ前処理の標準化**:

```python
def standardize_feature_format(features, target_shape=None):
    """特徴量フォーマットを標準化"""
    # NumPy配列に変換
    if not isinstance(features, np.ndarray):
        features = np.array(features)

    # 2次元配列に変換
    if len(features.shape) == 1:
        features = features.reshape(-1, 1)

    # 目標形状にリサイズ（必要に応じて）
    if target_shape and features.shape != target_shape:
        if features.shape[0] == target_shape[0]:
            # 特徴量数を調整
            if features.shape[1] < target_shape[1]:
                # パディング
                padding = np.zeros((features.shape[0], target_shape[1] - features.shape[1]))
                features = np.hstack([features, padding])
            elif features.shape[1] > target_shape[1]:
                # 切り詰め
                features = features[:, :target_shape[1]]

    return features
```

### 問題 2: データ品質の問題

**症状**:

- 予期しない結果
- 学習性能の低下
- 特徴量重要度の異常値

**診断ツール**:

```python
def comprehensive_data_quality_check(features, feature_names):
    """包括的なデータ品質チェック"""
    quality_report = {}

    for i, name in enumerate(feature_names):
        feature_data = features[:, i]

        quality_issues = []

        # 欠損値チェック
        missing_ratio = np.isnan(feature_data).mean()
        if missing_ratio > 0.05:
            quality_issues.append(f"高い欠損値率: {missing_ratio:.2%}")

        # 無限値チェック
        infinite_ratio = np.isinf(feature_data).mean()
        if infinite_ratio > 0:
            quality_issues.append(f"無限値を含む: {infinite_ratio:.2%}")

        # 分散チェック
        if np.var(feature_data) < 1e-8:
            quality_issues.append("分散がほぼゼロ（定数特徴量の可能性）")

        # 外れ値チェック
        Q1, Q3 = np.percentile(feature_data[~np.isnan(feature_data)], [25, 75])
        IQR = Q3 - Q1
        outlier_bounds = [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
        outlier_ratio = np.mean((feature_data < outlier_bounds[0]) | (feature_data > outlier_bounds[1]))
        if outlier_ratio > 0.1:
            quality_issues.append(f"多くの外れ値: {outlier_ratio:.2%}")

        # スケールチェック
        feature_range = np.ptp(feature_data[~np.isnan(feature_data)])
        if feature_range > 1000:
            quality_issues.append(f"大きなスケール範囲: {feature_range:.2e}")

        quality_report[name] = {
            'issues': quality_issues,
            'missing_ratio': missing_ratio,
            'infinite_ratio': infinite_ratio,
            'variance': np.var(feature_data),
            'range': feature_range,
            'outlier_ratio': outlier_ratio
        }

    return quality_report

# 使用例
quality_report = comprehensive_data_quality_check(features, feature_names)
for feature, report in quality_report.items():
    if report['issues']:
        print(f"{feature}: {', '.join(report['issues'])}")
```

**解決方法**:

1. **データクリーニングパイプライン**:

```python
def clean_feature_data(features, feature_names, cleaning_config=None):
    """特徴量データのクリーニング"""
    if cleaning_config is None:
        cleaning_config = {
            'handle_missing': 'median',
            'handle_infinite': 'clip',
            'handle_outliers': 'cap',
            'remove_constant': True
        }

    cleaned_features = features.copy()
    removed_features = []

    for i, name in enumerate(feature_names):
        feature_data = cleaned_features[:, i]

        # 欠損値処理
        if cleaning_config['handle_missing'] == 'median':
            median_val = np.nanmedian(feature_data)
            feature_data[np.isnan(feature_data)] = median_val
        elif cleaning_config['handle_missing'] == 'mean':
            mean_val = np.nanmean(feature_data)
            feature_data[np.isnan(feature_data)] = mean_val

        # 無限値処理
        if cleaning_config['handle_infinite'] == 'clip':
            feature_data[np.isinf(feature_data)] = np.nan
            # 再度欠損値処理
            median_val = np.nanmedian(feature_data)
            feature_data[np.isnan(feature_data)] = median_val

        # 外れ値処理
        if cleaning_config['handle_outliers'] == 'cap':
            Q1, Q3 = np.percentile(feature_data, [25, 75])
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            feature_data = np.clip(feature_data, lower_bound, upper_bound)

        # 定数特徴量の除去
        if cleaning_config['remove_constant'] and np.var(feature_data) < 1e-8:
            removed_features.append(name)
            continue

        cleaned_features[:, i] = feature_data

    # 除去された特徴量を削除
    if removed_features:
        keep_indices = [i for i, name in enumerate(feature_names) if name not in removed_features]
        cleaned_features = cleaned_features[:, keep_indices]
        cleaned_feature_names = [name for name in feature_names if name not in removed_features]
    else:
        cleaned_feature_names = feature_names

    return cleaned_features, cleaned_feature_names, removed_features
```

## パフォーマンス関連の問題

### 問題 1: GAT 訓練が収束しない

**症状**:

- 損失が減少しない
- 訓練が振動する
- 勾配爆発/消失

**診断**:

```python
def diagnose_gat_training_issues(training_history):
    """GAT訓練問題の診断"""
    losses = training_history['losses']

    diagnosis = {
        'convergence_issues': [],
        'recommendations': []
    }

    # 収束チェック
    if len(losses) > 10:
        recent_losses = losses[-10:]
        loss_variance = np.var(recent_losses)
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]

        if loss_variance > 0.01:
            diagnosis['convergence_issues'].append('損失が振動している')
            diagnosis['recommendations'].append('学習率を下げる')

        if loss_trend > 0.001:
            diagnosis['convergence_issues'].append('損失が増加傾向')
            diagnosis['recommendations'].append('勾配クリッピングを適用')

        if min(recent_losses) == max(recent_losses):
            diagnosis['convergence_issues'].append('損失が変化しない')
            diagnosis['recommendations'].append('学習率を上げるか、モデル構造を変更')

    # 勾配チェック
    if 'gradients' in training_history:
        grad_norms = training_history['gradients']
        if max(grad_norms) > 10:
            diagnosis['convergence_issues'].append('勾配爆発')
            diagnosis['recommendations'].append('勾配クリッピングを適用')
        if min(grad_norms) < 1e-6:
            diagnosis['convergence_issues'].append('勾配消失')
            diagnosis['recommendations'].append('学習率を上げるか、活性化関数を変更')

    return diagnosis
```

**解決方法**:

1. **学習率の調整**:

```python
# 学習率スケジューラーの使用
import torch.optim.lr_scheduler as lr_scheduler

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

# 訓練ループ内で
for epoch in range(num_epochs):
    loss = train_epoch(model, data, optimizer)
    scheduler.step(loss)
```

2. **勾配クリッピング**:

```python
# 勾配クリッピングの適用
max_grad_norm = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
```

3. **早期停止の実装**:

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

# 使用例
early_stopping = EarlyStopping(patience=20)
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_data, optimizer)
    val_loss = validate(model, val_data)

    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

### 問題 2: 特徴量選択が不安定

**症状**:

- 異なる実行で選択される特徴量が大きく変わる
- 交差検証での性能のばらつきが大きい

**解決方法**:

1. **安定性選択の実装**:

```python
def stable_feature_selection(features, labels, n_trials=50, stability_threshold=0.6):
    """安定した特徴量選択"""
    n_features = features.shape[1]
    selection_counts = np.zeros(n_features)

    for trial in range(n_trials):
        # ブートストラップサンプリング
        indices = np.random.choice(len(features), size=int(0.8 * len(features)), replace=False)
        bootstrap_features = features[indices]
        bootstrap_labels = labels[indices]

        # 特徴量選択
        selector = SelectKBest(f_classif, k=min(50, n_features))
        selector.fit(bootstrap_features, bootstrap_labels)

        # 選択された特徴量をカウント
        selected_mask = selector.get_support()
        selection_counts[selected_mask] += 1

    # 安定した特徴量を特定
    stability_scores = selection_counts / n_trials
    stable_features = stability_scores >= stability_threshold

    return stable_features, stability_scores
```

2. **アンサンブル特徴量選択**:

```python
def ensemble_feature_selection(features, labels, methods=['rfe', 'lasso', 'mutual_info']):
    """複数手法による特徴量選択のアンサンブル"""
    n_features = features.shape[1]
    selection_votes = np.zeros(n_features)

    for method in methods:
        if method == 'rfe':
            from sklearn.feature_selection import RFE
            from sklearn.ensemble import RandomForestClassifier
            selector = RFE(RandomForestClassifier(), n_features_to_select=50)
        elif method == 'lasso':
            from sklearn.feature_selection import SelectFromModel
            from sklearn.linear_model import LassoCV
            selector = SelectFromModel(LassoCV())
        elif method == 'mutual_info':
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            selector = SelectKBest(mutual_info_classif, k=50)

        selector.fit(features, labels)
        selected_mask = selector.get_support()
        selection_votes[selected_mask] += 1

    # 過半数で選択された特徴量を採用
    final_selection = selection_votes > len(methods) / 2

    return final_selection, selection_votes
```

## 特徴量分析の問題

### 問題 1: 重要度分析結果が直感と異なる

**症状**:

- 明らかに重要でない特徴量が高い重要度を示す
- 重要であるべき特徴量の重要度が低い

**診断**:

```python
def analyze_importance_anomalies(importance_scores, feature_names, domain_knowledge=None):
    """重要度異常の分析"""
    anomalies = []

    # 統計的異常値の検出
    Q1, Q3 = np.percentile(importance_scores, [25, 75])
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR

    for i, (score, name) in enumerate(zip(importance_scores, feature_names)):
        # 異常に高い重要度
        if score > outlier_threshold:
            anomalies.append({
                'feature': name,
                'importance': score,
                'type': 'unexpectedly_high',
                'investigation_needed': True
            })

        # ドメイン知識との照合（提供されている場合）
        if domain_knowledge:
            expected_importance = domain_knowledge.get(name, 'unknown')
            if expected_importance == 'high' and score < np.median(importance_scores):
                anomalies.append({
                    'feature': name,
                    'importance': score,
                    'type': 'unexpectedly_low',
                    'expected': expected_importance
                })

    return anomalies

def investigate_feature_importance(features, labels, feature_names, suspicious_features):
    """疑わしい特徴量の詳細調査"""
    investigation_results = {}

    for feature_name in suspicious_features:
        feature_idx = feature_names.index(feature_name)
        feature_data = features[:, feature_idx]

        # 基本統計
        stats = {
            'mean': np.mean(feature_data),
            'std': np.std(feature_data),
            'min': np.min(feature_data),
            'max': np.max(feature_data),
            'unique_values': len(np.unique(feature_data))
        }

        # ラベルとの関係
        correlation_with_label = np.corrcoef(feature_data, labels)[0, 1]

        # 分布の確認
        from scipy import stats as scipy_stats
        normality_p = scipy_stats.normaltest(feature_data)[1]

        investigation_results[feature_name] = {
            'basic_stats': stats,
            'label_correlation': correlation_with_label,
            'is_normal': normality_p > 0.05,
            'potential_issues': []
        }

        # 潜在的な問題の特定
        if stats['std'] < 1e-6:
            investigation_results[feature_name]['potential_issues'].append('ほぼ定数特徴量')

        if stats['unique_values'] < 5:
            investigation_results[feature_name]['potential_issues'].append('カテゴリカル特徴量の可能性')

        if abs(correlation_with_label) < 0.01:
            investigation_results[feature_name]['potential_issues'].append('ラベルとの相関が非常に低い')

    return investigation_results
```

**解決方法**:

1. **重要度計算方法の見直し**:

```python
def robust_importance_calculation(weights, feature_names, validation_method='permutation'):
    """より頑健な重要度計算"""

    if validation_method == 'permutation':
        # 順列重要度による検証
        from sklearn.inspection import permutation_importance
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier()
        model.fit(features, labels)

        perm_importance = permutation_importance(model, features, labels, n_repeats=10)
        validated_importance = perm_importance.importances_mean

    elif validation_method == 'shap':
        # SHAP値による解釈
        import shap

        model = RandomForestClassifier()
        model.fit(features, labels)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)
        validated_importance = np.mean(np.abs(shap_values), axis=0)

    # 元の重要度と比較
    correlation = np.corrcoef(np.abs(weights), validated_importance)[0, 1]

    return {
        'original_importance': weights,
        'validated_importance': validated_importance,
        'correlation': correlation,
        'consistent': correlation > 0.7
    }
```

### 問題 2: 相関分析で偽の相関が検出される

**症状**:

- 論理的に関係のない特徴量間で高い相関
- 多重比較による偽陽性

**解決方法**:

1. **多重比較補正**:

```python
def corrected_correlation_analysis(features, feature_names, alpha=0.05):
    """多重比較補正を適用した相関分析"""
    from scipy.stats import pearsonr
    from statsmodels.stats.multitest import multipletests

    n_features = len(feature_names)
    correlation_matrix = np.zeros((n_features, n_features))
    p_values = np.zeros((n_features, n_features))

    # 全ペアの相関を計算
    for i in range(n_features):
        for j in range(i+1, n_features):
            corr, p_val = pearsonr(features[:, i], features[:, j])
            correlation_matrix[i, j] = corr
            correlation_matrix[j, i] = corr
            p_values[i, j] = p_val
            p_values[j, i] = p_val

    # 多重比較補正
    upper_triangle_indices = np.triu_indices(n_features, k=1)
    upper_triangle_p_values = p_values[upper_triangle_indices]

    rejected, corrected_p_values, _, _ = multipletests(
        upper_triangle_p_values, alpha=alpha, method='fdr_bh'
    )

    # 補正されたp値を行列に戻す
    corrected_p_matrix = np.zeros((n_features, n_features))
    corrected_p_matrix[upper_triangle_indices] = corrected_p_values
    corrected_p_matrix = corrected_p_matrix + corrected_p_matrix.T

    # 有意な相関のみを保持
    significant_correlations = (corrected_p_matrix < alpha) & (np.abs(correlation_matrix) > 0.1)

    return {
        'correlation_matrix': correlation_matrix,
        'corrected_p_values': corrected_p_matrix,
        'significant_correlations': significant_correlations
    }
```

## GAT 最適化の問題

### 問題 1: 注意重みが均一になる

**症状**:

- すべてのエッジに対して似たような注意重み
- 注意機構が機能していない

**診断**:

```python
def diagnose_attention_uniformity(attention_weights, threshold=0.01):
    """注意重みの均一性を診断"""
    diagnosis = {}

    # 注意重みの分散を計算
    attention_variance = np.var(attention_weights, axis=1)
    uniform_heads = attention_variance < threshold

    diagnosis['uniform_attention_heads'] = np.sum(uniform_heads)
    diagnosis['total_heads'] = len(attention_variance)
    diagnosis['uniform_ratio'] = np.mean(uniform_heads)

    # エントロピーの計算
    def calculate_attention_entropy(weights):
        # 正規化
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        # エントロピー計算
        entropy = -np.sum(weights * np.log(weights + 1e-10), axis=1)
        return entropy

    entropies = calculate_attention_entropy(attention_weights)
    max_entropy = np.log(attention_weights.shape[1])  # 完全に均一な場合のエントロピー

    diagnosis['attention_entropies'] = entropies
    diagnosis['normalized_entropies'] = entropies / max_entropy
    diagnosis['high_entropy_ratio'] = np.mean(entropies > 0.9 * max_entropy)

    return diagnosis
```

**解決方法**:

1. **注意機構の改善**:

```python
class ImprovedGATLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features

        # 注意機構の正則化
        self.attention_dropout = torch.nn.Dropout(attention_dropout)

        # 温度パラメータ（注意の鋭さを制御）
        self.temperature = torch.nn.Parameter(torch.tensor(1.0))

        # 通常のGAT層
        self.gat_conv = GATConv(in_features, out_features, heads=num_heads, dropout=dropout)

    def forward(self, x, edge_index):
        # 温度スケーリングを適用
        x = self.gat_conv(x, edge_index)

        # 注意の多様性を促進する正則化項
        attention_weights = self.gat_conv.attention_weights
        entropy_loss = self.calculate_attention_entropy_loss(attention_weights)

        return x, entropy_loss

    def calculate_attention_entropy_loss(self, attention_weights):
        """注意の多様性を促進する損失"""
        # 各ヘッドの注意エントロピーを計算
        normalized_attention = F.softmax(attention_weights / self.temperature, dim=-1)
        entropy = -torch.sum(normalized_attention * torch.log(normalized_attention + 1e-10), dim=-1)

        # 高いエントロピーを促進（多様性向上）
        max_entropy = torch.log(torch.tensor(float(attention_weights.size(-1))))
        entropy_loss = -torch.mean(entropy / max_entropy)

        return entropy_loss
```

2. **注意正則化の追加**:

```python
def train_gat_with_attention_regularization(model, data, optimizer, epochs=200):
    """注意正則化を含むGAT訓練"""

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 順伝播
        out, entropy_loss = model(data.x, data.edge_index)

        # メイン損失
        main_loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        # 注意多様性損失
        diversity_loss = entropy_loss

        # 注意スパース性損失（過度な均一化を防ぐ）
        attention_weights = model.gat_conv.attention_weights
        sparsity_loss = torch.mean(torch.sum(attention_weights ** 2, dim=-1))

        # 総損失
        total_loss = main_loss + 0.01 * diversity_loss + 0.001 * sparsity_loss

        total_loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f'Epoch {epoch}: Main Loss: {main_loss:.4f}, '
                  f'Diversity Loss: {diversity_loss:.4f}, '
                  f'Sparsity Loss: {sparsity_loss:.4f}')
```

### 問題 2: GAT 埋め込みの品質が低い

**症状**:

- 下流タスクでの性能が悪い
- 埋め込み空間でのクラスタリング品質が低い

**診断**:

```python
def evaluate_embedding_quality(embeddings, labels, node_types=None):
    """埋め込み品質の包括的評価"""
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.cluster import KMeans

    quality_metrics = {}

    # 1. クラスタリング品質
    n_clusters = len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    quality_metrics['silhouette_score'] = silhouette_score(embeddings, labels)
    quality_metrics['clustering_ari'] = adjusted_rand_score(labels, cluster_labels)

    # 2. 分離性（クラス間距離 vs クラス内距離）
    within_class_distances = []
    between_class_distances = []

    for class_label in np.unique(labels):
        class_mask = labels == class_label
        class_embeddings = embeddings[class_mask]

        if len(class_embeddings) > 1:
            # クラス内距離
            class_center = np.mean(class_embeddings, axis=0)
            within_distances = np.linalg.norm(class_embeddings - class_center, axis=1)
            within_class_distances.extend(within_distances)

            # クラス間距離
            other_class_centers = []
            for other_label in np.unique(labels):
                if other_label != class_label:
                    other_mask = labels == other_label
                    other_center = np.mean(embeddings[other_mask], axis=0)
                    other_class_centers.append(other_center)

            for other_center in other_class_centers:
                between_distance = np.linalg.norm(class_center - other_center)
                between_class_distances.append(between_distance)

    quality_metrics['separability_ratio'] = (
        np.mean(between_class_distances) / np.mean(within_class_distances)
    )

    # 3. 次元利用効率
    pca = PCA()
    pca.fit(embeddings)
    explained_variance_ratio = pca.explained_variance_ratio_

    # 95%の分散を説明するのに必要な次元数
    cumsum_variance = np.cumsum(explained_variance_ratio)
    effective_dimensions = np.argmax(cumsum_variance >= 0.95) + 1

    quality_metrics['effective_dimensions'] = effective_dimensions
    quality_metrics['dimension_efficiency'] = effective_dimensions / embeddings.shape[1]

    return quality_metrics
```

**解決方法**:

1. **階層的な学習**:

```python
def hierarchical_gat_training(model, data, num_stages=3):
    """段階的なGAT訓練"""

    # Stage 1: 構造学習（グラフの構造を学習）
    print("Stage 1: 構造学習")
    structure_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(50):
        model.train()
        structure_optimizer.zero_grad()

        out = model(data.x, data.edge_index)

        # 構造保持損失（隣接ノードは似た埋め込みを持つべき）
        edge_embeddings = out[data.edge_index]
        structure_loss = F.mse_loss(edge_embeddings[0], edge_embeddings[1])

        structure_loss.backward()
        structure_optimizer.step()

    # Stage 2: 特徴学習（ノード特徴を学習）
    print("Stage 2: 特徴学習")
    feature_optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(100):
        model.train()
        feature_optimizer.zero_grad()

        out = model(data.x, data.edge_index)

        # 特徴再構成損失
        reconstructed_features = model.feature_decoder(out)  # デコーダーが必要
        feature_loss = F.mse_loss(reconstructed_features, data.x)

        feature_loss.backward()
        feature_optimizer.step()

    # Stage 3: タスク学習（最終タスクで微調整）
    print("Stage 3: タスク学習")
    task_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(200):
        model.train()
        task_optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        task_loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])

        task_loss.backward()
        task_optimizer.step()
```

## 設定関連の問題

### 問題 1: YAML 設定ファイルの構文エラー

**症状**:

```
yaml.scanner.ScannerError: mapping values are not allowed here
```

**解決方法**:

1. **YAML 構文の検証**:

```python
import yaml

def validate_yaml_config(config_path):
    """YAML設定ファイルの検証"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print("✓ YAML構文は正しいです")
        return config
    except yaml.YAMLError as e:
        print(f"✗ YAML構文エラー: {e}")

        # より詳細なエラー情報
        if hasattr(e, 'problem_mark'):
            mark = e.problem_mark
            print(f"エラー位置: 行 {mark.line + 1}, 列 {mark.column + 1}")

        return None

def suggest_yaml_fixes(config_path):
    """YAML修正案の提案"""
    with open(config_path, 'r') as file:
        lines = file.readlines()

    suggestions = []

    for i, line in enumerate(lines):
        line_num = i + 1

        # よくある問題をチェック
        if ':' in line and not line.strip().startswith('#'):
            # コロンの後にスペースがない
            if ':' in line and not ': ' in line and not ':\n' in line:
                suggestions.append(f"行 {line_num}: コロンの後にスペースを追加してください")

            # インデントの問題
            if len(line) - len(line.lstrip()) not in [0, 2, 4, 6, 8]:
                suggestions.append(f"行 {line_num}: インデントは2の倍数にしてください")

    return suggestions
```

2. **設定スキーマの検証**:

```python
from jsonschema import validate, ValidationError

def validate_config_schema(config, schema):
    """設定のスキーマ検証"""
    try:
        validate(instance=config, schema=schema)
        print("✓ 設定スキーマは正しいです")
        return True
    except ValidationError as e:
        print(f"✗ スキーマ検証エラー: {e.message}")
        print(f"エラーパス: {' -> '.join(str(p) for p in e.path)}")
        return False

# 基本的なスキーマ例
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "pipeline": {
            "type": "object",
            "properties": {
                "stages": {"type": "array", "items": {"type": "string"}},
                "enable_cache": {"type": "boolean"},
                "cache_expiry_hours": {"type": "integer", "minimum": 1}
            },
            "required": ["stages"]
        }
    },
    "required": ["pipeline"]
}
```

### 問題 2: 設定値の不整合

**症状**:

- パイプラインが期待通りに動作しない
- 一部の設定が無視される

**解決方法**:

1. **設定整合性チェック**:

```python
def check_config_consistency(config):
    """設定の整合性チェック"""
    issues = []

    # パイプライン設定チェック
    if 'pipeline' in config:
        pipeline_config = config['pipeline']

        # ステージ依存関係チェック
        stages = pipeline_config.get('stages', [])
        if 'optimization' in stages and 'design' not in stages:
            issues.append("最適化ステージには設計ステージが必要です")

        if 'gat_enhancement' in stages and 'optimization' not in stages:
            issues.append("GAT強化ステージには最適化ステージが必要です")

        # リソース設定チェック
        max_workers = pipeline_config.get('max_workers', 1)
        max_memory_mb = pipeline_config.get('max_memory_mb', 1024)

        if max_workers > 8 and max_memory_mb < 2048:
            issues.append("多数のワーカー使用時はメモリを増やすことを推奨")

    # 分析設定チェック
    if 'analysis' in config:
        analysis_config = config['analysis']

        # 重要度分析と相関分析の設定整合性
        if (analysis_config.get('importance_analysis', {}).get('enabled', False) and
            not analysis_config.get('correlation_analysis', {}).get('enabled', False)):
            issues.append("重要度分析を使用する場合は相関分析も有効化することを推奨")

    return issues
```

## よくある質問 (FAQ)

### Q1: システムをより高速化するにはどうすればよいですか？

**A1**: 以下の方法で高速化できます：

1. **並列処理の有効化**:

```yaml
pipeline:
  enable_parallel: true
  max_workers: 8 # CPUコア数に応じて調整
```

2. **キャッシュの活用**:

```yaml
pipeline:
  enable_cache: true
  cache_expiry_hours: 24
```

3. **不要な処理の無効化**:

```yaml
analysis:
  distribution_analysis:
    enabled: false # 不要な場合は無効化
```

4. **バッチサイズの調整**:

```python
# メモリに収まる最大サイズに設定
process_large_dataset_in_batches(features, batch_size=5000)
```

### Q2: メモリ使用量を削減するにはどうすればよいですか？

**A2**: メモリ使用量削減の方法：

1. **データ型の最適化**:

```python
# float64 → float32
features = features.astype(np.float32)
```

2. **バッチ処理の使用**:

```python
for batch in process_in_batches(data, batch_size=1000):
    result = process_batch(batch)
    # バッチごとに処理してメモリを解放
```

3. **疎行列の使用** (該当する場合):

```python
from scipy.sparse import csr_matrix
sparse_features = csr_matrix(features)
```

4. **不要なデータのクリーンアップ**:

```python
import gc
del unnecessary_data
gc.collect()
```

### Q3: 特徴量選択の結果が不安定です。どう対処すればよいですか？

**A3**: 安定性向上の方法：

1. **安定性選択の使用**:

```python
stable_features, stability_scores = stable_feature_selection(
    features, labels, n_trials=100, stability_threshold=0.7
)
```

2. **複数手法のアンサンブル**:

```python
final_selection, votes = ensemble_feature_selection(
    features, labels, methods=['rfe', 'lasso', 'mutual_info']
)
```

3. **クロスバリデーションでの検証**:

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, selected_features, labels, cv=10)
print(f"安定性: {np.std(scores):.3f}")
```

### Q4: GAT 最適化で良い結果が得られません。

**A4**: GAT 最適化の改善方法：

1. **段階的な次元探索**:

```python
# 粗い探索から細かい探索へ
coarse_dims = [16, 32, 64, 128]
fine_dims = range(best_coarse - 8, best_coarse + 9, 2)
```

2. **正則化の追加**:

```python
# 注意の多様性を促進
attention_entropy_loss = calculate_attention_entropy(attention_weights)
total_loss += 0.01 * attention_entropy_loss
```

3. **事前訓練の活用**:

```python
# グラフ構造を事前学習
pretrain_on_graph_structure(model, graph_data)
# その後、メインタスクで微調整
finetune_on_main_task(model, task_data)
```

### Q5: 大規模データセットでシステムがクラッシュします。

**A5**: 大規模データ対応方法：

1. **段階的処理**:

```python
# データを小さなチャンクに分割
for chunk in pd.read_csv('large_data.csv', chunksize=10000):
    process_chunk(chunk)
```

2. **メモリマッピング**:

```python
# メモリマップを使用した効率的な読み込み
data = np.memmap('large_array.dat', dtype=np.float32, mode='r')
```

3. **分散処理**:

```python
# Daskを使用した分散処理
import dask.dataframe as dd
df = dd.read_csv('large_data.csv')
result = df.map_partitions(process_function)
```

### Q6: 設定ファイルがうまく読み込まれません。

**A6**: 設定ファイルの問題解決：

1. **構文チェック**:

```bash
# YAMLの構文確認
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

2. **エンコーディング確認**:

```python
# UTF-8で保存されていることを確認
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
```

3. **パス確認**:

```python
import os
config_path = os.path.abspath('config.yaml')
print(f"設定ファイルパス: {config_path}")
print(f"ファイル存在: {os.path.exists(config_path)}")
```

## デバッグ手法

### 1. ログレベルの活用

```python
import logging

# デバッグ情報を詳細に出力
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 設定でログレベルを制御
pipeline:
  log_level: 'DEBUG'  # DEBUG, INFO, WARNING, ERROR
```

### 2. プロファイリング

```python
import cProfile
import pstats

def profile_function(func, *args, **kwargs):
    """関数のプロファイリング"""
    profiler = cProfile.Profile()
    profiler.enable()

    result = func(*args, **kwargs)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # 上位10件を表示

    return result

# 使用例
result = profile_function(analyzer.analyze_feature_importance, weights, feature_names)
```

### 3. メモリ使用量監視

```python
import psutil
import tracemalloc

def monitor_memory_usage(func):
    """メモリ使用量を監視するデコレータ"""
    def wrapper(*args, **kwargs):
        # メモリトレースを開始
        tracemalloc.start()

        # 実行前のメモリ使用量
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # 関数実行
        result = func(*args, **kwargs)

        # 実行後のメモリ使用量
        mem_after = process.memory_info().rss / 1024 / 1024  # MB

        # トレース統計
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"メモリ使用量: {mem_before:.1f}MB → {mem_after:.1f}MB "
              f"(変化: {mem_after - mem_before:+.1f}MB)")
        print(f"ピークメモリ: {peak / 1024 / 1024:.1f}MB")

        return result
    return wrapper

# 使用例
@monitor_memory_usage
def analyze_features(features, labels):
    return analyzer.analyze_feature_importance(features, labels)
```

### 4. データ整合性チェック

```python
def comprehensive_data_validation(features, labels, feature_names):
    """包括的なデータ検証"""
    validation_results = {
        'errors': [],
        'warnings': [],
        'info': []
    }

    # 基本形状チェック
    if features.shape[0] != len(labels):
        validation_results['errors'].append(
            f"特徴量のサンプル数 ({features.shape[0]}) とラベル数 ({len(labels)}) が一致しません"
        )

    if features.shape[1] != len(feature_names):
        validation_results['errors'].append(
            f"特徴量数 ({features.shape[1]}) と特徴量名数 ({len(feature_names)}) が一致しません"
        )

    # データ品質チェック
    nan_ratio = np.isnan(features).mean()
    if nan_ratio > 0.1:
        validation_results['warnings'].append(
            f"欠損値が多いです: {nan_ratio:.1%}"
        )

    inf_ratio = np.isinf(features).mean()
    if inf_ratio > 0:
        validation_results['errors'].append(
            f"無限値が含まれています: {inf_ratio:.1%}"
        )

    # 特徴量の分散チェック
    low_variance_features = []
    for i, name in enumerate(feature_names):
        if np.var(features[:, i]) < 1e-10:
            low_variance_features.append(name)

    if low_variance_features:
        validation_results['warnings'].append(
            f"低分散特徴量: {low_variance_features}"
        )

    return validation_results
```

## サポートとコミュニティ

### 問題報告の方法

1. **問題の詳細情報を収集**:

   - エラーメッセージの全文
   - 使用している設定ファイル
   - データの形状と型
   - 実行環境の情報

2. **再現可能な最小例を作成**:

```python
# 問題を再現する最小限のコード例
import numpy as np
from analysis.feature_analysis import FeatureImportanceAnalyzer

# ダミーデータで問題を再現
features = np.random.randn(100, 10)
weights = np.random.randn(10)
feature_names = [f'feature_{i}' for i in range(10)]

analyzer = FeatureImportanceAnalyzer()
result = analyzer.analyze_feature_importance(weights, feature_names)
# ここでエラーが発生
```

3. **システム情報の収集**:

```python
import sys
import numpy as np
import pandas as pd
import sklearn

print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
```

### コミュニティリソース

- **ドキュメント**: このガイドと API 仕様書
- **設定例**: `/configs/` ディレクトリの設定ファイル
- **テストスイート**: `/tests/` ディレクトリのテストケース
- **サンプルコード**: 各モジュールの使用例

### 貢献方法

1. **バグ報告**: 上記の方法で詳細な報告を作成
2. **機能要求**: 具体的な使用ケースと期待される動作を説明
3. **コード貢献**: テストケースを含むプルリクエスト
4. **ドキュメント改善**: 不明確な箇所の指摘と改善案

このトラブルシューティングガイドを参照しても問題が解決しない場合は、できるだけ詳細な情報とともに問題を報告してください。コミュニティによるサポートを受けることができます。
