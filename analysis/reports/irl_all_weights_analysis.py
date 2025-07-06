#!/usr/bin/env python3
"""
IRL全重み詳細分析 - 全ての特徴量重みを詳細に表示・分析
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("src")


def get_all_feature_names_and_descriptions():
    """全特徴量名と説明の対応を取得"""

    # 基本特徴量（25次元）
    base_features = [
        ("task_days_since_last_activity", "タスクの最終活動からの日数"),
        ("task_discussion_activity", "タスクのディスカッション活動度"),
        ("task_text_length", "タスクテキストの長さ"),
        ("task_code_block_count", "タスク内のコードブロック数"),
        ("task_label_bug", "バグラベルの有無"),
        ("task_label_enhancement", "機能強化ラベルの有無"),
        ("task_label_documentation", "ドキュメントラベルの有無"),
        ("task_label_question", "質問ラベルの有無"),
        ("task_label_help wanted", "ヘルプ募集ラベルの有無"),
        ("dev_recent_activity_count", "開発者の最近の活動数"),
        ("dev_current_workload", "開発者の現在の作業負荷"),
        ("dev_total_lines_changed", "開発者の総変更行数"),
        ("dev_collaboration_network_size", "開発者の協力ネットワークサイズ"),
        ("dev_comment_interactions", "開発者のコメント相互作用数"),
        ("dev_cross_issue_activity", "開発者のクロスイシュー活動"),
        ("match_collaborated_with_task_author", "タスク作成者との協力履歴"),
        ("match_collaborator_overlap_count", "共通協力者数"),
        ("match_has_prior_collaboration", "過去の協力関係の有無"),
        ("match_skill_intersection_count", "スキル交差数"),
        ("match_file_experience_count", "ファイル経験数"),
        ("match_affinity_for_bug", "バグ対応への親和性"),
        ("match_affinity_for_enhancement", "機能強化への親和性"),
        ("match_affinity_for_documentation", "ドキュメント作業への親和性"),
        ("match_affinity_for_question", "質問対応への親和性"),
        ("match_affinity_for_help wanted", "ヘルプ対応への親和性"),
    ]

    # GAT特徴量（37次元）
    gat_features = [
        ("gat_similarity", "GAT類似度"),
        ("gat_dev_expertise", "GAT開発者専門性"),
        ("gat_task_popularity", "GATタスク人気度"),
        ("gat_collaboration_strength", "GAT協力関係強度"),
        ("gat_network_centrality", "GATネットワーク中心性"),
    ]
    
    # GAT埋め込み（32次元）
    for i in range(32):
        gat_features.append((f"gat_dev_emb_{i}", f"GAT開発者埋め込み第{i}次元"))

    return base_features + gat_features


def load_irl_weights():
    """IRL重みを読み込み"""
    weight_files = [
        "data/learned_weights_training.npy",
        "reward_weights.npy",
        "learned_reward_weights.npy",
        "data/learned_reward_weights.npy",
        "data/reward_weights.npy"
    ]
    
    for weight_file in weight_files:
        if Path(weight_file).exists():
            try:
                weights = np.load(weight_file)
                print(f"✅ IRL重みを読み込み: {weight_file}")
                print(f"   重み数: {len(weights)}")
                return weights, weight_file
            except Exception as e:
                print(f"❌ {weight_file} の読み込みエラー: {e}")
    
    print("❌ IRL重みファイルが見つかりません")
    return None, None


def analyze_all_weights(weights, feature_descriptions):
    """全重みの詳細分析"""
    print(f"\n" + "="*80)
    print("📊 IRL全重み詳細分析")
    print("="*80)
    
    # 基本統計
    print(f"\n【基本統計】")
    print(f"重み数: {len(weights)}")
    print(f"平均値: {np.mean(weights):.6f}")
    print(f"標準偏差: {np.std(weights):.6f}")
    print(f"最小値: {np.min(weights):.6f}")
    print(f"最大値: {np.max(weights):.6f}")
    print(f"絶対値の平均: {np.mean(np.abs(weights)):.6f}")
    
    # ゼロ重みの数
    zero_weights = np.sum(np.abs(weights) < 1e-6)
    print(f"ほぼゼロの重み数: {zero_weights}/{len(weights)} ({zero_weights/len(weights)*100:.1f}%)")
    
    # 正負の分布
    positive_weights = np.sum(weights > 0)
    negative_weights = np.sum(weights < 0)
    print(f"正の重み: {positive_weights} ({positive_weights/len(weights)*100:.1f}%)")
    print(f"負の重み: {negative_weights} ({negative_weights/len(weights)*100:.1f}%)")


def create_complete_weight_table(weights, feature_descriptions, output_dir="outputs"):
    """全重みの完全テーブルを作成"""
    print(f"\n【全重み詳細テーブル】")
    
    # データフレーム作成
    data = []
    for i, (feature_name, description) in enumerate(feature_descriptions):
        if i < len(weights):
            weight_val = weights[i]
            abs_weight = abs(weight_val)
            sign = "+" if weight_val > 0 else "-" if weight_val < 0 else "0"
            
            data.append({
                "番号": i,
                "特徴量名": feature_name,
                "説明": description,
                "重み値": weight_val,
                "絶対値": abs_weight,
                "符号": sign,
                "重要度ランク": 0  # 後で設定
            })
    
    df = pd.DataFrame(data)
    
    # 重要度ランクを設定
    df = df.sort_values("絶対値", ascending=False)
    df["重要度ランク"] = range(1, len(df) + 1)
    df = df.sort_values("番号")  # 元の順序に戻す
    
    # テーブル表示
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    
    print(df.to_string(index=False, float_format='%.6f'))
    
    # CSVファイルとして保存
    import os
    os.makedirs(output_dir, exist_ok=True)
    csv_path = f"{output_dir}/irl_all_weights_table.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n✅ 全重みテーブルをCSVで保存: {csv_path}")
    
    return df


def analyze_by_feature_category(weights, feature_descriptions):
    """特徴量カテゴリ別の分析"""
    print(f"\n【カテゴリ別分析】")
    
    # カテゴリ分け
    categories = {
        "タスク特徴量": [],
        "開発者特徴量": [],
        "マッチング特徴量": [],
        "GAT統計特徴量": [],
        "GAT埋め込み": []
    }
    
    for i, (feature_name, description) in enumerate(feature_descriptions):
        if i >= len(weights):
            continue
            
        weight_val = weights[i]
        
        if feature_name.startswith("task_"):
            categories["タスク特徴量"].append((feature_name, weight_val, description))
        elif feature_name.startswith("dev_"):
            categories["開発者特徴量"].append((feature_name, weight_val, description))
        elif feature_name.startswith("match_"):
            categories["マッチング特徴量"].append((feature_name, weight_val, description))
        elif feature_name.startswith("gat_") and not "emb" in feature_name:
            categories["GAT統計特徴量"].append((feature_name, weight_val, description))
        elif "gat_dev_emb" in feature_name:
            categories["GAT埋め込み"].append((feature_name, weight_val, description))
    
    # カテゴリ別統計
    for category_name, features in categories.items():
        if not features:
            continue
            
        weights_in_category = [w for _, w, _ in features]
        print(f"\n📁 {category_name} ({len(features)}個)")
        print(f"   平均重み: {np.mean(weights_in_category):.6f}")
        print(f"   重み範囲: [{np.min(weights_in_category):.6f}, {np.max(weights_in_category):.6f}]")
        print(f"   絶対値平均: {np.mean(np.abs(weights_in_category)):.6f}")
        
        # 上位3つの重み
        sorted_features = sorted(features, key=lambda x: abs(x[1]), reverse=True)
        print(f"   重要な特徴量（上位3つ）:")
        for i, (fname, weight, desc) in enumerate(sorted_features[:3]):
            print(f"     {i+1}. {fname}: {weight:.6f} ({desc})")


def create_comprehensive_visualizations(weights, feature_descriptions, output_dir="outputs"):
    """包括的な可視化"""
    print(f"\n【可視化生成中】")
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 全重みのバープロット
    plt.figure(figsize=(20, 12))
    
    # サブプロット1: 全重みのバープロット
    plt.subplot(3, 2, 1)
    indices = range(len(weights))
    colors = ['red' if w < 0 else 'blue' if w > 0 else 'gray' for w in weights]
    plt.bar(indices, weights, color=colors, alpha=0.7)
    plt.title('全IRL重み')
    plt.xlabel('特徴量インデックス')
    plt.ylabel('重み値')
    plt.grid(True, alpha=0.3)
    
    # サブプロット2: 絶対値重みのバープロット
    plt.subplot(3, 2, 2)
    abs_weights = np.abs(weights)
    plt.bar(indices, abs_weights, alpha=0.7)
    plt.title('重み絶対値')
    plt.xlabel('特徴量インデックス')
    plt.ylabel('重み絶対値')
    plt.grid(True, alpha=0.3)
    
    # サブプロット3: 重みのヒストグラム
    plt.subplot(3, 2, 3)
    plt.hist(weights, bins=30, alpha=0.7, edgecolor='black')
    plt.title('重み分布')
    plt.xlabel('重み値')
    plt.ylabel('頻度')
    plt.grid(True, alpha=0.3)
    
    # サブプロット4: 上位20重要特徴量
    plt.subplot(3, 2, 4)
    sorted_indices = np.argsort(np.abs(weights))[-20:]
    top_weights = weights[sorted_indices]
    top_labels = [f"{i}:{feature_descriptions[i][0][:15]}" for i in sorted_indices]
    
    colors = ['red' if w < 0 else 'blue' for w in top_weights]
    plt.barh(range(len(top_weights)), top_weights, color=colors, alpha=0.7)
    plt.yticks(range(len(top_weights)), top_labels, fontsize=8)
    plt.title('重要特徴量 Top20')
    plt.xlabel('重み値')
    plt.grid(True, alpha=0.3)
    
    # サブプロット5: カテゴリ別平均重み
    plt.subplot(3, 2, 5)
    category_weights = {}
    
    for i, (feature_name, description) in enumerate(feature_descriptions):
        if i >= len(weights):
            continue
            
        if feature_name.startswith("task_"):
            category = "タスク"
        elif feature_name.startswith("dev_"):
            category = "開発者"
        elif feature_name.startswith("match_"):
            category = "マッチング"
        elif feature_name.startswith("gat_") and not "emb" in feature_name:
            category = "GAT統計"
        elif "gat_dev_emb" in feature_name:
            category = "GAT埋め込み"
        else:
            category = "その他"
        
        if category not in category_weights:
            category_weights[category] = []
        category_weights[category].append(abs(weights[i]))
    
    categories = list(category_weights.keys())
    avg_weights = [np.mean(category_weights[cat]) for cat in categories]
    
    plt.bar(categories, avg_weights, alpha=0.7)
    plt.title('カテゴリ別平均重み絶対値')
    plt.xlabel('カテゴリ')
    plt.ylabel('平均重み絶対値')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # サブプロット6: 累積重要度
    plt.subplot(3, 2, 6)
    sorted_abs_weights = np.sort(np.abs(weights))[::-1]
    cumsum_weights = np.cumsum(sorted_abs_weights)
    cumsum_ratio = cumsum_weights / cumsum_weights[-1]
    
    plt.plot(range(1, len(cumsum_ratio) + 1), cumsum_ratio, 'o-', markersize=2)
    plt.title('累積重要度')
    plt.xlabel('特徴量数')
    plt.ylabel('累積重要度比率')
    plt.grid(True, alpha=0.3)
    
    # 80%ライン
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80%')
    plt.legend()
    
    plt.tight_layout()
    
    # 画像保存
    plot_path = f"{output_dir}/irl_all_weights_comprehensive_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 包括的可視化を保存: {plot_path}")


def generate_summary_report(weights, feature_descriptions, df, output_dir="outputs"):
    """総合レポート生成"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = f"{output_dir}/irl_all_weights_summary_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("IRL全重み詳細分析レポート\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # 基本統計
        f.write("【基本統計】\n")
        f.write(f"重み数: {len(weights)}\n")
        f.write(f"平均値: {np.mean(weights):.6f}\n")
        f.write(f"標準偏差: {np.std(weights):.6f}\n")
        f.write(f"最小値: {np.min(weights):.6f}\n")
        f.write(f"最大値: {np.max(weights):.6f}\n")
        f.write(f"絶対値の平均: {np.mean(np.abs(weights)):.6f}\n\n")
        
        # 上位重要特徴量
        f.write("【最重要特徴量 Top20】\n")
        top_20 = df.sort_values("絶対値", ascending=False).head(20)
        for _, row in top_20.iterrows():
            f.write(f"{row['重要度ランク']:2d}. {row['特徴量名']:<30} | {row['重み値']:>10.6f} | {row['説明']}\n")
        
        f.write("\n【最も正の重み Top10】\n")
        top_positive = df.sort_values("重み値", ascending=False).head(10)
        for _, row in top_positive.iterrows():
            f.write(f"    {row['特徴量名']:<30} | {row['重み値']:>10.6f} | {row['説明']}\n")
        
        f.write("\n【最も負の重み Top10】\n")
        top_negative = df.sort_values("重み値", ascending=True).head(10)
        for _, row in top_negative.iterrows():
            f.write(f"    {row['特徴量名']:<30} | {row['重み値']:>10.6f} | {row['説明']}\n")
        
        # カテゴリ別統計をレポートに追加
        f.write("\n【カテゴリ別統計】\n")
        categories = {}
        for _, row in df.iterrows():
            feature_name = row['特徴量名']
            if feature_name.startswith("task_"):
                category = "タスク特徴量"
            elif feature_name.startswith("dev_"):
                category = "開発者特徴量"
            elif feature_name.startswith("match_"):
                category = "マッチング特徴量"
            elif feature_name.startswith("gat_") and not "emb" in feature_name:
                category = "GAT統計特徴量"
            elif "gat_dev_emb" in feature_name:
                category = "GAT埋め込み"
            else:
                category = "その他"
            
            if category not in categories:
                categories[category] = []
            categories[category].append(row['重み値'])
        
        for category, cat_weights in categories.items():
            f.write(f"\n{category} ({len(cat_weights)}個):\n")
            f.write(f"  平均重み: {np.mean(cat_weights):.6f}\n")
            f.write(f"  重み範囲: [{np.min(cat_weights):.6f}, {np.max(cat_weights):.6f}]\n")
            f.write(f"  絶対値平均: {np.mean(np.abs(cat_weights)):.6f}\n")
    
    print(f"✅ 総合レポートを保存: {report_path}")


def main():
    """メイン実行関数"""
    print("🔍 IRL全重み詳細分析")
    
    # IRL重み読み込み
    weights, weight_file = load_irl_weights()
    if weights is None:
        return
    
    # 特徴量名と説明を取得
    feature_descriptions = get_all_feature_names_and_descriptions()
    
    # 次元数の確認
    if len(weights) != len(feature_descriptions):
        print(f"⚠️  重み数({len(weights)})と特徴量数({len(feature_descriptions)})が一致しません")
        min_len = min(len(weights), len(feature_descriptions))
        weights = weights[:min_len]
        feature_descriptions = feature_descriptions[:min_len]
        print(f"   最初の{min_len}次元で分析を実行します")
    
    # 全重み分析
    analyze_all_weights(weights, feature_descriptions)
    
    # 完全テーブル作成
    df = create_complete_weight_table(weights, feature_descriptions)
    
    # カテゴリ別分析
    analyze_by_feature_category(weights, feature_descriptions)
    
    # 包括的可視化
    create_comprehensive_visualizations(weights, feature_descriptions)
    
    # 総合レポート生成
    generate_summary_report(weights, feature_descriptions, df)
    
    print("\n✅ IRL全重み詳細分析完了！")
    print("📁 生成されたファイル:")
    print("   - outputs/irl_all_weights_table.csv")
    print("   - outputs/irl_all_weights_comprehensive_analysis.png")
    print("   - outputs/irl_all_weights_summary_report.txt")


if __name__ == "__main__":
    main()
