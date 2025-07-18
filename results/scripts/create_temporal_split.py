#!/usr/bin/env python3
"""
時系列データ分割スクリプト
データリークを防ぐため、時系列順にデータを分割
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


def filter_by_year_range(data, start_year, end_year):
    """指定された年の範囲でデータをフィルタリング"""
    filtered = []
    for item in data:
        created_at = item["created_at"]
        year = int(created_at[:4])
        if start_year <= year <= end_year:
            filtered.append(item)
    return filtered


def main():
    parser = argparse.ArgumentParser(description="時系列データ分割")
    parser.add_argument(
        "--input", default="data/backlog.json", help="入力データファイル"
    )
    parser.add_argument("--irl-start", type=int, default=2019, help="IRL学習開始年")
    parser.add_argument("--irl-end", type=int, default=2021, help="IRL学習終了年")
    parser.add_argument("--rl-start", type=int, default=2022, help="RL訓練開始年")
    parser.add_argument("--rl-end", type=int, default=2022, help="RL訓練終了年")
    parser.add_argument("--test-start", type=int, default=2023, help="テスト開始年")
    parser.add_argument("--test-end", type=int, default=2023, help="テスト終了年")

    args = parser.parse_args()

    print("🔄 時系列データ分割開始")
    print("=" * 60)

    # データ読み込み
    print(f"📖 データ読み込み: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"   総データ数: {len(data):,} タスク")

    # IRL用データ（エキスパート軌跡作成用）
    irl_data = filter_by_year_range(data, args.irl_start, args.irl_end)
    print(f"\n🧠 IRL学習用データ ({args.irl_start}-{args.irl_end}年):")
    print(f"   タスク数: {len(irl_data):,}")

    # RL訓練用データ
    rl_data = filter_by_year_range(data, args.rl_start, args.rl_end)
    print(f"\n🤖 RL訓練用データ ({args.rl_start}-{args.rl_end}年):")
    print(f"   タスク数: {len(rl_data):,}")

    # テスト用データ
    test_data = filter_by_year_range(data, args.test_start, args.test_end)
    print(f"\n🎯 テスト用データ ({args.test_start}-{args.test_end}年):")
    print(f"   タスク数: {len(test_data):,}")

    # ファイル保存
    output_dir = Path("data")

    # IRL用（expert trajectories作成用）
    irl_path = output_dir / "backlog_irl.json"
    with open(irl_path, "w", encoding="utf-8") as f:
        json.dump(irl_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ IRL用データ保存: {irl_path}")

    # RL訓練用
    rl_path = output_dir / "backlog_training_new.json"
    with open(rl_path, "w", encoding="utf-8") as f:
        json.dump(rl_data, f, indent=2, ensure_ascii=False)
    print(f"✅ RL訓練用データ保存: {rl_path}")

    # テスト用
    test_path = output_dir / "backlog_test_2023.json"
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    print(f"✅ テスト用データ保存: {test_path}")

    # サマリー表示
    total_split = len(irl_data) + len(rl_data) + len(test_data)
    print(f"\n📊 データ分割サマリー:")
    print(
        f"   IRL用:    {len(irl_data):,} タスク ({len(irl_data)/total_split*100:.1f}%)"
    )
    print(f"   RL訓練用: {len(rl_data):,} タスク ({len(rl_data)/total_split*100:.1f}%)")
    print(
        f"   テスト用: {len(test_data):,} タスク ({len(test_data)/total_split*100:.1f}%)"
    )
    print(f"   総計:     {total_split:,} タスク")

    # 設定ファイル更新の提案
    print(f"\n🔧 設定ファイル更新が必要:")
    print(f"   configs/unified_rl.yaml を以下に変更:")
    print(f"   env:")
    print(f'     backlog_path: "data/backlog_training_new.json"')
    print(f'     expert_trajectories_path: "data/expert_trajectories_new.pkl"')

    print(f"\n⚠️ 次のステップ:")
    print(f"   1. IRL用データでexpert trajectories再生成")
    print(f"   2. RL訓練用データで強化学習実行")
    print(f"   3. テスト用データで最終評価")


if __name__ == "__main__":
    main()
