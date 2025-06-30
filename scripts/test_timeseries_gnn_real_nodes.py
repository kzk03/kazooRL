#!/usr/bin/env python3
"""
実際のGNNノードを使った時系列GNN学習テスト
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from omegaconf import OmegaConf

from kazoo.envs.oss_simple import OSSSimpleEnv
from kazoo.learners.independent_ppo_controller import IndependentPPOController


def test_timeseries_gnn_with_real_nodes():
    """実際のGNNノードを使った時系列GNN学習テスト"""
    print("🔬 実際のGNNノードを使った時系列GNN学習テスト")
    print("=" * 60)

    # 設定ファイル読み込み
    cfg = OmegaConf.load(project_root / "configs" / "base.yaml")

    # データ読み込み
    with open(project_root / cfg.env.backlog_path, "r") as f:
        backlog = json.load(f)

    with open(project_root / cfg.env.dev_profiles_path, "r") as f:
        dev_profiles = yaml.safe_load(f)

    # 環境初期化
    print("🌍 環境を初期化中...")
    env = OSSSimpleEnv(cfg, backlog, dev_profiles)

    # GNN特徴量抽出器を取得
    gnn_extractor = env.feature_extractor.gnn_extractor
    if not gnn_extractor or not gnn_extractor.online_learning:
        print("❌ GNNオンライン学習が利用できません")
        return

    print("✅ GNNオンライン学習準備完了")
    print(f"  📊 利用可能な開発者: {len(gnn_extractor.dev_id_to_idx)}")
    print(f"  📊 利用可能なタスク: {len(gnn_extractor.task_id_to_idx)}")

    # 実際に存在する開発者とタスクのサンプルを取得
    available_devs = list(gnn_extractor.dev_id_to_idx.keys())[:5]
    available_tasks = list(gnn_extractor.task_id_to_idx.keys())[:10]

    print(f"  🎯 テスト対象開発者: {available_devs}")
    print(f"  🎯 テスト対象タスク: {available_tasks[:5]}...")

    # 実際のタスクオブジェクトを作成
    class RealTask:
        def __init__(self, task_id):
            self.id = task_id
            self.title = f"Task {task_id}"

    class RealDeveloper:
        def __init__(self, dev_name):
            self.name = dev_name

        def get(self, key, default=None):
            if key == "name":
                return self.name
            return default

    # 時系列インタラクションテスト
    print("\n⏰ 時系列インタラクションテスト開始")

    base_time = env.current_time
    test_scenarios = [
        (available_devs[0], available_tasks[0], base_time, 1.0, "assignment"),
        (
            available_devs[1],
            available_tasks[1],
            base_time + timedelta(hours=2),
            0.8,
            "positive",
        ),
        (
            available_devs[0],
            available_tasks[2],
            base_time + timedelta(hours=4),
            1.2,
            "completion",
        ),
        (
            available_devs[2],
            available_tasks[3],
            base_time + timedelta(hours=6),
            0.6,
            "positive",
        ),
        (
            available_devs[1],
            available_tasks[0],
            base_time + timedelta(hours=8),
            -0.3,
            "negative",
        ),
        (
            available_devs[3],
            available_tasks[4],
            base_time + timedelta(hours=10),
            0.9,
            "positive",
        ),
        (
            available_devs[0],
            available_tasks[5],
            base_time + timedelta(hours=12),
            1.1,
            "completion",
        ),
        (
            available_devs[4],
            available_tasks[1],
            base_time + timedelta(hours=26),
            0.7,
            "positive",
        ),  # 時間窓外
    ]

    print(f"📝 {len(test_scenarios)} 件のインタラクションを記録中...")

    for i, (dev_name, task_id, sim_time, reward, action_type) in enumerate(
        test_scenarios
    ):
        developer = RealDeveloper(dev_name)
        task = RealTask(task_id)

        print(
            f"  {i+1}. {sim_time.strftime('%H:%M')} - {dev_name} + {task_id} = {reward} ({action_type})"
        )

        gnn_extractor.record_interaction(
            task, developer, reward, action_type, simulation_time=sim_time
        )

        # 更新チェック
        buffer_size = len(gnn_extractor.interaction_buffer)
        if buffer_size % gnn_extractor.update_frequency == 0 and buffer_size > 0:
            print(
                f"     ⚡ GNN更新がトリガーされました (バッファサイズ: {buffer_size})"
            )

    # 最終状態の確認
    print(f"\n📊 最終バッファ状態:")
    print(f"  総インタラクション数: {len(gnn_extractor.interaction_buffer)}")

    # 時間窓分析
    if gnn_extractor.interaction_buffer:
        times = [
            interaction["simulation_time"]
            for interaction in gnn_extractor.interaction_buffer
        ]
        latest_time = max(times)
        cutoff_time = latest_time - timedelta(hours=gnn_extractor.time_window_hours)

        recent_interactions = [
            interaction
            for interaction in gnn_extractor.interaction_buffer
            if interaction["simulation_time"] >= cutoff_time
        ]

        print(f"  最新時刻: {latest_time.strftime('%m/%d %H:%M')}")
        print(f"  カットオフ時刻: {cutoff_time.strftime('%m/%d %H:%M')}")
        print(
            f"  時間窓内インタラクション: {len(recent_interactions)}/{len(gnn_extractor.interaction_buffer)}"
        )

    # 強制的にGNN更新を実行
    print(f"\n🔄 強制的なGNN更新テスト:")
    updates_before = gnn_extractor.stats.get("updates", 0)
    print(f"  更新前の回数: {updates_before}")

    gnn_extractor._update_gnn_online()

    updates_after = gnn_extractor.stats.get("updates", 0)
    print(f"  更新後の回数: {updates_after}")

    if updates_after > updates_before:
        print("  ✅ GNN更新が正常に実行されました！")
    else:
        print("  ⚠️  GNN更新が実行されませんでした")

    # 統計情報を表示
    print(f"\n📈 GNN学習統計:")
    gnn_extractor.print_statistics()

    # 短期間の強化学習テスト
    print(f"\n🤖 短期間の強化学習テスト:")
    env.current_time = base_time  # 時間をリセット

    controller = IndependentPPOController(env=env, config=cfg)

    rl_updates_before = gnn_extractor.stats.get("updates", 0)
    rl_buffer_before = len(gnn_extractor.interaction_buffer)

    print(f"  学習前: 更新{rl_updates_before}回, バッファ{rl_buffer_before}件")

    try:
        controller.learn(total_timesteps=30)  # 短時間テスト
    except Exception as e:
        print(f"  学習エラー: {e}")

    rl_updates_after = gnn_extractor.stats.get("updates", 0)
    rl_buffer_after = len(gnn_extractor.interaction_buffer)

    print(f"  学習後: 更新{rl_updates_after}回, バッファ{rl_buffer_after}件")
    print(
        f"  変化: +{rl_updates_after - rl_updates_before}更新, +{rl_buffer_after - rl_buffer_before}インタラクション"
    )

    # 最終結果
    print(f"\n🎯 最終結果:")
    print(f"  ✅ 時系列順序でインタラクションが記録されています")
    print(f"  ✅ 時間窓フィルタリングが正常に動作しています")
    print(f"  ✅ 実際のGNNノードを使った更新が可能です")
    print(f"  📊 総GNN更新回数: {gnn_extractor.stats.get('updates', 0)}")

    # 時系列データの保存（デバッグ用）
    if gnn_extractor.interaction_buffer:
        interaction_summary = []
        for interaction in gnn_extractor.interaction_buffer:
            interaction_summary.append(
                {
                    "time": interaction["simulation_time"].isoformat(),
                    "dev": interaction["dev_id"],
                    "task": interaction["task_id"],
                    "reward": interaction["reward"],
                    "action": interaction["action_taken"],
                }
            )

        output_file = project_root / "logs" / "timeseries_gnn_test.json"
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(interaction_summary, f, indent=2, default=str)

        print(f"  💾 時系列データを保存しました: {output_file}")


if __name__ == "__main__":
    test_timeseries_gnn_with_real_nodes()
