#!/usr/bin/env python3
"""
特徴量分析基盤の使用例
====================

実装した特徴量分析クラスの実際の使用方法を示します。
"""

import sys
from pathlib import Path

import numpy as np

# プロジェクトルートを追加
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
sys.path.append(str(Path(__file__).parent))

from feature_correlation_analyzer import FeatureCorrelationAnalyzer
from feature_distribution_analyzer import FeatureDistributionAnalyzer
from feature_importance_analyzer import FeatureImportanceAnalyzer


def load_real_irl_data():
    """実際のIRLデータを読み込み（利用可能な場合）"""
    data_dir = project_root / "kazoo" / "data"
    
    # IRL重みファイルを探す
    weights_files = [
        "learned_weights_bot_excluded.npy",
        "learned_weights_test.npy"
    ]
    
    weights_path = None
    for weights_file in weights_files:
        candidate_path = data_dir / weights_file
        if candidate_path.exists():
            weights_path = candidate_path
            break
    
    if weights_path is None:
        print("⚠️  実際のIRL重みファイルが見つかりません。サンプルデータを使用します。")
        return None, None, None
    
    # 重みを読み込み
    weights = np.load(weights_path)
    print(f"✅ IRL重み読み込み成功: {weights_path.name} ({len(weights)}次元)")
    
    # 特徴量名を取得（FeatureExtractorから）
    try:
        from omegaconf import OmegaConf

        from kazoo.src.kazoo.features.feature_extractor import FeatureExtractor

        # 設定を作成
        cfg = OmegaConf.create({
            "features": {
                "all_labels": ["bug", "enhancement", "documentation", "question", "help wanted"],
                "label_to_skills": {
                    "bug": ["debugging", "analysis"],
                    "enhancement": ["python", "design"],
                    "documentation": ["writing"],
                    "question": ["communication"],
                    "help wanted": ["collaboration"]
                }
            },
            "irl": {"use_gat": True}
        })
        
        feature_extractor = FeatureExtractor(cfg)
        feature_names = feature_extractor.feature_names
        
        # 次元数を合わせる
        if len(feature_names) != len(weights):
            min_dim = min(len(feature_names), len(weights))
            feature_names = feature_names[:min_dim]
            weights = weights[:min_dim]
        
        print(f"✅ 特徴量名取得成功: {len(feature_names)}個")
        
        # サンプル特徴量データを生成（実際のデータがない場合）
        n_samples = 1000
        feature_data = generate_realistic_feature_data(feature_names, n_samples)
        
        return weights, feature_names, feature_data
        
    except Exception as e:
        print(f"⚠️  特徴量名取得に失敗: {e}")
        return None, None, None


def generate_realistic_feature_data(feature_names, n_samples=1000):
    """現実的な特徴量データを生成"""
    n_features = len(feature_names)
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
    
    return features


def demonstrate_feature_analysis():
    """特徴量分析のデモンストレーション"""
    print("🚀 特徴量分析基盤デモンストレーション")
    print("="*80)
    
    # データを読み込み
    weights, feature_names, feature_data = load_real_irl_data()
    
    if weights is None:
        # サンプルデータを使用
        print("📊 サンプルデータを生成中...")
        n_samples, n_features = 1000, 25
        feature_names = [
            # タスク特徴量
            "task_days_since_last_activity", "task_discussion_activity", 
            "task_text_length", "task_code_block_count",
            "task_label_bug", "task_label_enhancement",
            # 開発者特徴量
            "dev_recent_activity_count", "dev_current_workload",
            "dev_total_lines_changed", "dev_collaboration_network_size",
            "dev_comment_interactions", "dev_cross_issue_activity",
            # マッチング特徴量
            "match_collaborated_with_task_author", "match_collaborator_overlap_count",
            "match_has_prior_collaboration", "match_skill_intersection_count",
            "match_file_experience_count", "match_affinity_for_bug",
            "match_affinity_for_enhancement",
            # GAT特徴量
            "gat_similarity", "gat_dev_expertise"
        ]
        
        # 残りをGAT埋め込みで埋める
        while len(feature_names) < n_features:
            feature_names.append(f"feature_{len(feature_names) - 21}")
        
        feature_data = generate_realistic_feature_data(feature_names, n_samples)
        weights = np.random.randn(n_features) * 0.5
    
    print(f"📈 データ概要:")
    print(f"   - サンプル数: {feature_data.shape[0]:,}")
    print(f"   - 特徴量数: {len(feature_names)}")
    print(f"   - IRL重み範囲: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
    
    # 出力ディレクトリ作成
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # 重みファイルを一時保存
    weights_path = "temp_weights.npy"
    np.save(weights_path, weights)
    
    try:
        # 1. 特徴量重要度分析
        print("\n" + "="*60)
        print("📊 特徴量重要度分析")
        print("="*60)
        
        importance_analyzer = FeatureImportanceAnalyzer(weights_path, feature_names)
        importance_results = importance_analyzer.analyze_importance()
        
        print("🏆 重要度ランキング TOP10:")
        for i, (name, weight, importance) in enumerate(importance_results['importance_ranking'][:10], 1):
            print(f"  {i:2d}. {name:35s} | 重み:{weight:8.5f} | 重要度:{importance:8.5f}")
        
        print("\n📋 カテゴリ別重要度:")
        for category, stats in importance_results['category_importance'].items():
            print(f"  {category:20s}: 平均重要度 {stats['mean_importance']:.5f} ({stats['count']}個)")
        
        # レポート生成
        importance_report = importance_analyzer.generate_importance_report(
            output_dir / "importance_analysis_report.txt"
        )
        print(f"✅ 重要度分析レポート生成: {importance_report}")
        
        # 2. 特徴量相関分析
        print("\n" + "="*60)
        print("🔗 特徴量相関分析")
        print("="*60)
        
        correlation_analyzer = FeatureCorrelationAnalyzer(feature_data, feature_names)
        correlation_results = correlation_analyzer.analyze_correlations(correlation_threshold=0.6)
        
        corr_stats = correlation_results['correlation_statistics']
        print(f"📊 相関統計:")
        print(f"   - 平均絶対相関: {corr_stats['mean_abs_correlation']:.4f}")
        print(f"   - 高相関ペア(|r|≥0.6): {corr_stats['high_corr_count_06']}個")
        print(f"   - 低相関ペア(|r|≤0.2): {corr_stats['low_corr_count_02']}個")
        
        if correlation_results['high_correlation_pairs']:
            print("\n🔍 高相関ペア TOP5:")
            for i, (feat1, feat2, corr) in enumerate(correlation_results['high_correlation_pairs'][:5], 1):
                print(f"  {i}. {feat1[:25]:25s} - {feat2[:25]:25s} | r={corr:6.3f}")
        
        if correlation_results['redundant_features']:
            print(f"\n⚠️  冗長特徴量候補: {len(correlation_results['redundant_features'])}個")
            for item in correlation_results['redundant_features'][:3]:
                print(f"   削除候補: {item['redundant_feature'][:30]:30s} (相関: {item['correlation']:.3f})")
        
        # レポート生成
        correlation_report = correlation_analyzer.generate_correlation_report(
            output_dir / "correlation_analysis_report.txt"
        )
        print(f"✅ 相関分析レポート生成: {correlation_report}")
        
        # 3. 特徴量分布分析
        print("\n" + "="*60)
        print("📈 特徴量分布分析")
        print("="*60)
        
        distribution_analyzer = FeatureDistributionAnalyzer(feature_data, feature_names)
        distribution_results = distribution_analyzer.analyze_distributions()
        
        scale_info = distribution_results['scale_imbalance']
        print(f"⚖️  スケール不均衡:")
        print(f"   - 標準偏差比（最大/最小）: {scale_info['std_ratio_max_min']:.2f}")
        print(f"   - レンジ比（最大/最小）: {scale_info['range_ratio_max_min']:.2f}")
        print(f"   - 深刻なスケール不均衡: {'はい' if scale_info['scale_imbalance_severe'] else 'いいえ'}")
        
        quality_issues = distribution_results['data_quality_issues']
        print(f"\n🚨 データ品質問題:")
        total_issues = sum(len(features) for features in quality_issues.values())
        if total_issues > 0:
            for issue_type, features_list in quality_issues.items():
                if features_list:
                    print(f"   - {issue_type}: {len(features_list)}個")
        else:
            print("   - 重大な品質問題は検出されませんでした")
        
        # 正規性検定結果
        normality_tests = distribution_results['normality_tests']
        normal_count = sum(1 for test in normality_tests.values() if test.get('is_normal_shapiro', False))
        print(f"\n📊 正規性検定:")
        print(f"   - 正規分布に従う特徴量: {normal_count}/{len(normality_tests)}")
        
        # レポート生成
        distribution_report = distribution_analyzer.generate_distribution_report(
            output_dir / "distribution_analysis_report.txt"
        )
        print(f"✅ 分布分析レポート生成: {distribution_report}")
        
        # 4. 統合分析サマリー
        print("\n" + "="*80)
        print("📋 統合分析サマリー")
        print("="*80)
        
        print("🎯 主要な発見:")
        
        # 最重要特徴量
        top_feature = importance_results['importance_ranking'][0]
        print(f"   - 最重要特徴量: {top_feature[0]} (重要度: {top_feature[2]:.5f})")
        
        # 基本特徴量 vs GAT特徴量
        comparison = importance_results['basic_vs_gat_comparison']
        if 'error' not in comparison:
            basic_importance = comparison['basic_features']['mean_importance']
            gat_importance = comparison['gat_features']['mean_importance']
            print(f"   - 基本特徴量平均重要度: {basic_importance:.5f}")
            print(f"   - GAT特徴量平均重要度: {gat_importance:.5f}")
            
            if basic_importance > gat_importance:
                print("   → 基本特徴量の方が重要度が高い傾向")
            else:
                print("   → GAT特徴量の方が重要度が高い傾向")
        
        # 相関問題
        high_corr_count = len(correlation_results['high_correlation_pairs'])
        if high_corr_count > 0:
            print(f"   - 高相関ペア: {high_corr_count}個 → 冗長性の可能性")
        
        # スケール問題
        if scale_info['scale_imbalance_severe']:
            print("   - 深刻なスケール不均衡 → 正規化が必要")
        
        print("\n💡 推奨アクション:")
        
        # 冗長特徴量の削除
        if correlation_results['redundant_features']:
            print(f"   1. {len(correlation_results['redundant_features'])}個の冗長特徴量候補の削除を検討")
        
        # スケーリング
        if scale_info['std_ratio_max_min'] > 10:
            print("   2. 特徴量の標準化またはmin-max正規化を実施")
        
        # 分布の改善
        extreme_skew_count = sum(1 for stats in distribution_results['distribution_stats'].values() 
                               if abs(stats.get('skewness', 0)) > 2)
        if extreme_skew_count > 0:
            print(f"   3. {extreme_skew_count}個の極端に歪んだ特徴量の対数変換を検討")
        
        print(f"\n📁 生成されたレポート:")
        print(f"   - {importance_report}")
        print(f"   - {correlation_report}")
        print(f"   - {distribution_report}")
        
        print("\n🎉 特徴量分析基盤デモンストレーション完了！")
        
    finally:
        # クリーンアップ
        Path(weights_path).unlink(missing_ok=True)


if __name__ == "__main__":
    demonstrate_feature_analysis()