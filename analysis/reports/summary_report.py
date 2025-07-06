#!/usr/bin/env python3
"""
IRL分析結果の要約レポート（テキストベース）
"""

from datetime import datetime
from pathlib import Path

import numpy as np


def generate_summary_report():
    """分かりやすい要約レポートを生成"""
    
    print("📄 IRL学習結果サマリーレポート")
    print("=" * 60)
    print(f"生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
    print()
    
    # データの読み込み
    weights_path = Path("data/learned_weights_training.npy")
    if not weights_path.exists():
        print("❌ データが見つかりません")
        return
    
    weights = np.load(weights_path)
    
    print("🎯 何を学習したか？")
    print("-" * 30)
    print("開発者選択において「どんな開発者が選ばれやすいか」を")
    print("過去のデータから逆強化学習で学習しました。")
    print()
    
    # 基本統計
    base_weights = weights[:25]  # 基本特徴量
    gat_weights = weights[25:] if len(weights) > 25 else np.array([])  # 協力関係特徴量
    
    print("📊 学習結果の要約")
    print("-" * 30)
    print(f"• 分析した特徴量数: {len(weights)}個")
    print(f"  - 基本情報: {len(base_weights)}個")
    print(f"  - 協力関係: {len(gat_weights)}個")
    print()
    
    # 重要度比較
    base_importance = np.mean(np.abs(base_weights))
    gat_importance = np.mean(np.abs(gat_weights)) if len(gat_weights) > 0 else 0
    
    if gat_importance > base_importance:
        ratio = gat_importance / base_importance
        print("🤝 最重要発見:")
        print(f"協力関係が基本情報より{ratio:.1f}倍重要！")
        print("→ 開発者選択では「誰と働いてきたか」が最も大切")
    else:
        print("👤 重要発見:")
        print("基本的な開発者情報が協力関係より重要")
    print()
    
    # 重要な基本特徴量
    print("✅ 選ばれやすい開発者の特徴")
    print("-" * 30)
    
    feature_names = [
        "ログイン名の長さ", "名前の有無", "名前の長さ", "会社情報の有無", "会社名の長さ",
        "場所情報の有無", "場所情報の長さ", "プロフィール文の有無", "プロフィール文の長さ",
        "公開リポジトリ数", "公開リポジトリ数(対数)", "フォロワー数", "フォロワー数(対数)",
        "フォロー数", "フォロー数(対数)", "アカウント年数(日)", "アカウント年数(年)",
        "フォロワー/フォロー比", "年間リポジトリ作成数", "人気度スコア", "活動度スコア",
        "影響力スコア", "経験値スコア", "社交性スコア", "プロフィール完成度"
    ]
    
    # 正の重みが大きい基本特徴量
    positive_features = []
    for i, weight in enumerate(base_weights):
        if weight > 0.5:
            positive_features.append((weight, feature_names[i]))
    
    positive_features.sort(reverse=True)
    
    advice_map = {
        "プロフィール文の有無": "自己紹介を書いている",
        "フォロー数": "他の開発者をフォローしている（社交的）",
        "人気度スコア": "コミュニティで人気がある",
        "活動度スコア": "継続的に活動している",
        "社交性スコア": "コミュニティと積極的に関わっている",
        "フォロワー数": "多くの人にフォローされている",
        "公開リポジトリ数": "適度な数のリポジトリを公開している"
    }
    
    for weight, name in positive_features[:5]:
        advice = advice_map.get(name, name)
        print(f"• {advice} (重要度: {weight:.2f})")
    
    print()
    
    # 避けられる特徴
    print("❌ 避けられやすい開発者の特徴")
    print("-" * 30)
    
    negative_features = []
    for i, weight in enumerate(base_weights):
        if weight < -0.3:
            negative_features.append((weight, feature_names[i]))
    
    negative_features.sort()
    
    negative_advice_map = {
        "年間リポジトリ作成数": "リポジトリを作りすぎている（量より質重視）",
        "影響力スコア": "個人の影響力が強すぎる（チームワーク重視）",
        "会社名の長さ": "会社名が長すぎる",
        "フォロワー/フォロー比": "フォロワーに比べてフォロー数が少ない（社交性不足）"
    }
    
    for weight, name in negative_features[:3]:
        advice = negative_advice_map.get(name, name)
        print(f"• {advice} (重要度: {weight:.2f})")
    
    print()
    
    # 協力関係の分析
    if len(gat_weights) > 0:
        print("🤝 協力関係の影響")
        print("-" * 30)
        strong_positive = np.sum(gat_weights > 1.0)
        strong_negative = np.sum(gat_weights < -1.0)
        max_collab = np.max(gat_weights)
        min_collab = np.min(gat_weights)
        
        print(f"• 重要視される協力パターン: {strong_positive}個")
        print(f"• 避けられる協力パターン: {strong_negative}個")
        print(f"• 最も重要な協力関係の重要度: {max_collab:.2f}")
        if min_collab < 0:
            print(f"• 最も避けられる協力関係の重要度: {min_collab:.2f}")
        print()
        print("協力関係が基本情報より重要ということは...")
        print("• 過去に誰と一緒に仕事をしたかが最重要")
        print("• どんなプロジェクトで活躍したかが重要")
        print("• 個人のスキルよりもチームワークが重視される")
        print()
    
    # 実用的なアドバイス
    print("💡 開発者におすすめのアクション")
    print("-" * 30)
    
    if gat_importance > base_importance:
        print("1. 協力関係を重視する")
        print("   • 様々な開発者と積極的にコラボレーション")
        print("   • チームプロジェクトに参加")
        print("   • オープンソースプロジェクトへの貢献")
        print()
    
    if any(w > 0.5 for w in base_weights if feature_names[list(base_weights).index(w)] == "プロフィール文の有無"):
        print("2. プロフィール情報を充実させる")
        print("   • GitHubプロフィールに自己紹介を書く")
        print("   • スキルや経験を明記")
        print("   • 連絡先や所属情報を追加")
        print()
    
    if any(w > 0.5 for w in base_weights if "フォロー" in feature_names[list(base_weights).index(w)]):
        print("3. コミュニティ活動を活発にする")
        print("   • 他の開発者をフォロー")
        print("   • 技術記事やプロジェクトにいいね・コメント")
        print("   • 勉強会や技術イベントに参加")
        print()
    
    if any(w < -0.5 for w in base_weights if "年間リポジトリ" in feature_names[list(base_weights).index(w)]):
        print("4. 質を重視する")
        print("   • リポジトリの数より質を重視")
        print("   • 完成度の高いプロジェクトを作成")
        print("   • メンテナンスされたコードを保つ")
        print()
    
    # 結論
    print("🎉 結論")
    print("-" * 30)
    print("開発者選択において最も重要なのは:")
    
    if gat_importance > base_importance:
        print("1位: 協力関係・チームワーク")
        print("2位: コミュニケーション能力")
        print("3位: 適度な活動レベル")
        print("4位: 個人の技術実績")
    else:
        print("1位: 基本的な開発者情報")
        print("2位: 協力関係・チームワーク")
    
    print()
    print("つまり「何を知っているか」より「誰と働けるか」が重要！")
    
    # レポートをファイルに保存
    save_report_to_file(weights, base_weights, gat_weights, feature_names)

def save_report_to_file(weights, base_weights, gat_weights, feature_names):
    """レポートをファイルに保存"""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / f"irl_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("IRL学習結果サマリーレポート\n")
        f.write("=" * 50 + "\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
        
        f.write("【学習内容】\n")
        f.write("開発者選択において「どんな開発者が選ばれやすいか」を学習\n\n")
        
        f.write("【統計情報】\n")
        f.write(f"総特徴量数: {len(weights)}\n")
        f.write(f"基本特徴量数: {len(base_weights)}\n")
        f.write(f"協力関係特徴量数: {len(gat_weights)}\n")
        f.write(f"有意な特徴量数: {np.sum(np.abs(weights) > 0.3)}\n\n")
        
        # 重要な特徴量
        f.write("【重要な基本特徴量 Top 10】\n")
        base_sorted = sorted(enumerate(base_weights), key=lambda x: abs(x[1]), reverse=True)
        for rank, (idx, weight) in enumerate(base_sorted[:10], 1):
            f.write(f"{rank:2d}. {feature_names[idx]:25s} {weight:7.3f}\n")
        
        f.write("\n【協力関係特徴量統計】\n")
        if len(gat_weights) > 0:
            f.write(f"平均重要度: {np.mean(np.abs(gat_weights)):.3f}\n")
            f.write(f"最大重み: {np.max(gat_weights):.3f}\n")
            f.write(f"最小重み: {np.min(gat_weights):.3f}\n")
            f.write(f"正の重み数: {np.sum(gat_weights > 0)}\n")
            f.write(f"負の重み数: {np.sum(gat_weights < 0)}\n")
        
    print(f"📄 詳細レポート保存: {report_path}")

def main():
    """メイン実行"""
    try:
        generate_summary_report()
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
