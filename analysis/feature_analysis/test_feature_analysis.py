#!/usr/bin/env python3
"""
特徴量分析基盤のテストスクリプト
==============================

実装した特徴量分析クラスの動作確認を行います。
"""

import sys
from pathlib import Path

import numpy as np

# プロジェクトルートを追加
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from kazoo.analysis.feature_analysis import (
    FeatureCorrelationAnalyzer,
    FeatureDistributionAnalyzer,
    FeatureImportanceAnalyzer,
)


def generate_test_data(n_samples=1000, n_features=25):
    """テスト用の特徴量データを生成"""
    print(f"テストデータ生成中... ({n_samples}サンプル, {n_features}特徴量)")
    
    # 特徴量名を定義
    feature_names = [
        # タスク特徴量
        "task_days_since_last_activity",
        "task_discussion_activity", 
        "task_text_length",
        "task_code_block_count",
        "task_label_bug",
        "task_label_enhancement",
        # 開発者特徴量
        "dev_recent_activity_count",
        "dev_current_workload",
        "dev_total_lines_changed",
        "dev_collaboration_network_size",
        "dev_comment_interactions",
        "dev_cross_issue_activity",
        # マッチング特徴量
        "match_collaborated_with_task_author",
        "match_collaborator_overlap_count",
        "match_has_prior_collaboration",
        "match_skill_intersection_count",
        "match_file_experience_count",
        "match_affinity_for_bug",
        "match_affinity_for_enhancement",
        # GAT特徴量
        "gat_similarity",
        "gat_dev_expertise",
        "feature_0",
        "feature_1", 
        "feature_2",
        "feature_3"
    ]
    
    # 実際の特徴量数に合わせる
    if len(feature_names) > n_features:
        feature_names = feature_names[:n_features]
    elif len(feature_names) < n_features:
        for i in range(len(feature_names), n_features):
            feature_names.append(f"feature_{i}")
    
    # 現実的な特徴量データを生成
    features = np.zeros((n_samples, n_features))
    
    for i, name in enumerate(feature_names):
        if "days_since" in name:
            # 指数分布 (0-365日)
            features[:, i] = np.random.exponential(30, n_samples)
            features[:, i] = np.clip(features[:, i], 0, 365)
        elif "activity" in name or "count" in name:
            # ポアソン分布
            features[:, i] = np.random.poisson(3, n_samples)
        elif "length" in name:
            # ログノーマル分布
            features[:, i] = np.random.lognormal(4, 1, n_samples)
        elif "label_" in name or "collaborated" in name or "has_prior" in name:
            # バイナリ特徴量
            features[:, i] = np.random.binomial(1, 0.15, n_samples).astype(float)
        elif "workload" in name:
            # ガンマ分布
            features[:, i] = np.random.gamma(2, 2, n_samples)
            features[:, i] = np.clip(features[:, i], 0, 15)
        elif "lines_changed" in name:
            # ログノーマル分布
            features[:, i] = np.random.lognormal(3, 1.5, n_samples)
        elif "network" in name or "collaboration" in name:
            # べき乗分布
            features[:, i] = np.random.pareto(1.16, n_samples) * 2
            features[:, i] = np.clip(features[:, i], 0, 50)
        elif "affinity" in name or "similarity" in name:
            # ベータ分布 (0-1)
            features[:, i] = np.random.beta(1, 4, n_samples)
        elif name.startswith("feature_"):
            # GAT埋め込み: 標準正規分布
            features[:, i] = np.random.normal(0, 0.5, n_samples)
        else:
            # その他: 軽いランダムウォーク
            features[:, i] = np.random.normal(0, 0.8, n_samples)
    
    # IRL重みを生成
    weights = np.random.randn(n_features) * 0.5
    
    print("✅ テストデータ生成完了")
    return features, feature_names, weights


def test_feature_importance_analyzer():
    """FeatureImportanceAnalyzerのテスト"""
    print("\n" + "="*60)
    print("🧪 FeatureImportanceAnalyzer テスト開始")
    print("="*60)
    
    # テストデータ生成
    features, feature_names, weights = generate_test_data()
    
    # 重みファイルを一時保存
    weights_path = "test_weights.npy"
    np.save(weights_path, weights)
    
    try:
        # アナライザー初期化
        analyzer = FeatureImportanceAnalyzer(weights_path, feature_names)
        
        # 重要度分析実行
        results = analyzer.analyze_importance()
        
        print("✅ 重要度分析完了")
        print(f"   - 重要度ランキング: {len(results['importance_ranking'])}項目")
        print(f"   - カテゴリ別分析: {len(results['category_importance'])}カテゴリ")
        print(f"   - 統計的有意性: {len(results['statistical_significance'])}項目")
        
        # TOP5を表示
        print("\n重要度TOP5:")
        for i, (name, weight, importance) in enumerate(results['importance_ranking'][:5], 1):
            print(f"  {i}. {name:30s} | 重み:{weight:7.4f} | 重要度:{importance:7.4f}")
        
        # レポート生成テスト
        report_path = analyzer.generate_importance_report("test_importance_report.txt")
        print(f"✅ レポート生成完了: {report_path}")
        
        # 可視化テスト
        fig_path = analyzer.visualize_importance("outputs", show_plot=False)
        print(f"✅ 可視化完了: {fig_path}")
        
    finally:
        # 一時ファイル削除
        Path(weights_path).unlink(missing_ok=True)
    
    print("✅ FeatureImportanceAnalyzer テスト完了")


def test_feature_correlation_analyzer():
    """FeatureCorrelationAnalyzerのテスト"""
    print("\n" + "="*60)
    print("🧪 FeatureCorrelationAnalyzer テスト開始")
    print("="*60)
    
    # テストデータ生成
    features, feature_names, _ = generate_test_data()
    
    # アナライザー初期化
    analyzer = FeatureCorrelationAnalyzer(features, feature_names)
    
    # 相関分析実行
    results = analyzer.analyze_correlations(correlation_threshold=0.3)  # 閾値を下げてテスト
    
    print("✅ 相関分析完了")
    print(f"   - 相関行列: {results['correlation_matrix'].shape}")
    print(f"   - 高相関ペア: {len(results['high_correlation_pairs'])}ペア")
    print(f"   - 冗長特徴量候補: {len(results['redundant_features'])}個")
    print(f"   - カテゴリ間相関: {len(results['category_correlations'])}項目")
    
    # 高相関ペアTOP5を表示
    if results['high_correlation_pairs']:
        print("\n高相関ペアTOP5:")
        for i, (feat1, feat2, corr) in enumerate(results['high_correlation_pairs'][:5], 1):
            print(f"  {i}. {feat1[:20]:20s} - {feat2[:20]:20s} | r={corr:6.3f}")
    
    # レポート生成テスト
    report_path = analyzer.generate_correlation_report("test_correlation_report.txt")
    print(f"✅ レポート生成完了: {report_path}")
    
    # 可視化テスト
    fig_path = analyzer.visualize_correlations("outputs", show_plot=False)
    print(f"✅ 可視化完了: {fig_path}")
    
    print("✅ FeatureCorrelationAnalyzer テスト完了")


def test_feature_distribution_analyzer():
    """FeatureDistributionAnalyzerのテスト"""
    print("\n" + "="*60)
    print("🧪 FeatureDistributionAnalyzer テスト開始")
    print("="*60)
    
    # テストデータ生成
    features, feature_names, _ = generate_test_data()
    
    # アナライザー初期化
    analyzer = FeatureDistributionAnalyzer(features, feature_names)
    
    # 分布分析実行
    results = analyzer.analyze_distributions()
    
    print("✅ 分布分析完了")
    print(f"   - 分布統計: {len(results['distribution_stats'])}特徴量")
    print(f"   - 正規性検定: {len(results['normality_tests'])}特徴量")
    print(f"   - 外れ値検出: {len(results['outlier_detection'])}特徴量")
    print(f"   - カテゴリ別分布: {len(results['category_distributions'])}カテゴリ")
    
    # データ品質問題を表示
    quality_issues = results['data_quality_issues']
    print("\nデータ品質問題:")
    for issue_type, features_list in quality_issues.items():
        if features_list:
            print(f"  - {issue_type}: {len(features_list)}個")
    
    # スケール不均衡を表示
    scale_imbalance = results['scale_imbalance']
    print(f"\nスケール不均衡:")
    print(f"  - 標準偏差比（最大/最小）: {scale_imbalance['std_ratio_max_min']:.2f}")
    print(f"  - レンジ比（最大/最小）: {scale_imbalance['range_ratio_max_min']:.2f}")
    
    # レポート生成テスト
    report_path = analyzer.generate_distribution_report("test_distribution_report.txt")
    print(f"✅ レポート生成完了: {report_path}")
    
    # 可視化テスト
    fig_path = analyzer.visualize_distributions("outputs", show_plot=False, max_features=12)
    print(f"✅ 可視化完了: {fig_path}")
    
    print("✅ FeatureDistributionAnalyzer テスト完了")


def main():
    """メイン関数"""
    print("🚀 特徴量分析基盤テスト開始")
    print("="*80)
    
    # 出力ディレクトリ作成
    Path("outputs").mkdir(exist_ok=True)
    
    try:
        # 各アナライザーのテスト実行
        test_feature_importance_analyzer()
        test_feature_correlation_analyzer()
        test_feature_distribution_analyzer()
        
        print("\n" + "="*80)
        print("🎉 全テスト完了！")
        print("="*80)
        
        print("\n生成されたファイル:")
        for file_path in Path(".").glob("test_*.txt"):
            print(f"  - {file_path}")
        
        output_dir = Path("outputs")
        if output_dir.exists():
            for file_path in output_dir.glob("*.png"):
                print(f"  - {file_path}")
        
    except Exception as e:
        print(f"\n❌ テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())