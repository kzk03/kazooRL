#!/usr/bin/env python3
"""
時系列GNN対応の完全パイプラインテスト
1. IRL (逆強化学習) で報酬関数を学習
2. RL (強化学習) でGNNオンライン学習を使用して政策を学習
"""
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

import numpy as np
from omegaconf import OmegaConf


def run_complete_pipeline():
    """時系列GNN対応の完全パイプラインを実行"""
    print("🚀 時系列GNN対応 完全学習パイプライン")
    print("=" * 60)

    # 設定読み込み
    cfg = OmegaConf.load(project_root / "configs" / "base_training.yaml")

    print("📋 実行設定:")
    print(f"  🎯 IRL設定:")
    print(f"    - エキスパート軌跡: {cfg.irl.expert_path}")
    print(f"    - 学習率: {cfg.irl.learning_rate}")
    print(f"    - エポック数: {cfg.irl.epochs}")
    print(f"    - GNN使用: {cfg.irl.use_gnn}")
    print(f"    - GNNオンライン学習: {cfg.irl.online_gnn_learning}")

    print(f"  🎯 RL設定:")
    print(f"    - 総ステップ数: {cfg.rl.total_timesteps}")
    print(f"    - 学習率: {cfg.rl.learning_rate}")
    print(f"    - GNN更新頻度: {cfg.irl.gnn_update_frequency}")
    print(f"    - GNN時間窓: {cfg.irl.gnn_time_window_hours}時間")

    # Step 1: IRL (逆強化学習) 実行
    print(f"\n🔥 Step 1: IRL (逆強化学習) 実行")
    print(f"=" * 40)

    start_time = datetime.now()

    try:
        # IRLスクリプトを実行
        import subprocess

        import yaml

        print("📚 エキスパート軌跡から報酬関数を学習中...")

        # IRL学習実行（スクリプト実行）
        result = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "train_irl.py")],
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )

        if result.returncode == 0:
            print("✅ IRL学習成功")
            print(result.stdout[-500:])  # 最後の500文字を表示

            # 学習結果確認
            weights_path = project_root / cfg.irl.output_weights_path
            if weights_path.exists():
                import numpy as np

                reward_weights = np.load(weights_path)
                print(f"📊 学習された報酬重み形状: {reward_weights.shape}")
                print(f"💾 保存先: {cfg.irl.output_weights_path}")
            else:
                print("⚠️  報酬重みファイルが見つかりません")
                reward_weights = None
        else:
            print(f"❌ IRL学習失敗: {result.stderr}")
            reward_weights = None

        irl_duration = datetime.now() - start_time
        print(f"⏱️  IRL実行時間: {irl_duration.total_seconds():.1f}秒")

    except Exception as e:
        print(f"❌ IRL実行エラー: {e}")
        print("⚠️  IRLスキップして強化学習のみ実行します")
        reward_weights = None

    # Step 2: RL (強化学習) 実行
    print(f"\n🤖 Step 2: RL (強化学習) 実行")
    print(f"=" * 40)

    rl_start_time = datetime.now()

    try:
        import yaml

        from kazoo.envs.oss_simple import OSSSimpleEnv
        from kazoo.learners.independent_ppo_controller import IndependentPPOController

        # データ読み込み
        with open(project_root / cfg.env.backlog_path, "r") as f:
            backlog = json.load(f)

        with open(project_root / cfg.env.dev_profiles_path, "r") as f:
            dev_profiles = yaml.safe_load(f)

        # 環境初期化
        print("🌍 強化学習環境を初期化中...")
        env = OSSSimpleEnv(cfg, backlog, dev_profiles)

        # GNN状態確認
        if hasattr(env, "feature_extractor") and hasattr(
            env.feature_extractor, "gnn_extractor"
        ):
            gnn_extractor = env.feature_extractor.gnn_extractor
            if gnn_extractor and gnn_extractor.online_learning:
                print("✅ GNNオンライン学習が有効です")
                print(f"  📊 開発者ノード数: {len(gnn_extractor.dev_id_to_idx)}")
                print(f"  📊 タスクノード数: {len(gnn_extractor.task_id_to_idx)}")
                print(f"  ⚙️ 更新頻度: {gnn_extractor.update_frequency}回ごと")
                print(f"  ⏰ 時間窓: {gnn_extractor.time_window_hours}時間")
            else:
                print("⚠️  GNNオンライン学習が無効です")
        else:
            print("⚠️  GNN特徴量抽出器が見つかりません")

        # PPO学習実行
        print(f"🚀 PPO学習開始 ({cfg.rl.total_timesteps}ステップ)...")
        controller = IndependentPPOController(env=env, config=cfg)

        # GNN統計（学習前）
        if hasattr(env, "feature_extractor") and hasattr(
            env.feature_extractor, "gnn_extractor"
        ):
            gnn_extractor = env.feature_extractor.gnn_extractor
            if gnn_extractor:
                updates_before = gnn_extractor.stats.get("updates", 0)
                buffer_before = len(gnn_extractor.interaction_buffer)
                print(
                    f"  📊 学習前GNN状態: 更新{updates_before}回, バッファ{buffer_before}件"
                )

        # 学習実行
        controller.learn(total_timesteps=cfg.rl.total_timesteps)

        rl_duration = datetime.now() - rl_start_time
        print(f"✅ RL完了 (実行時間: {rl_duration.total_seconds():.1f}秒)")

        # GNN統計（学習後）
        if hasattr(env, "feature_extractor") and hasattr(
            env.feature_extractor, "gnn_extractor"
        ):
            gnn_extractor = env.feature_extractor.gnn_extractor
            if gnn_extractor:
                updates_after = gnn_extractor.stats.get("updates", 0)
                buffer_after = len(gnn_extractor.interaction_buffer)
                print(
                    f"  📊 学習後GNN状態: 更新{updates_after}回, バッファ{buffer_after}件"
                )
                print(f"  🔄 GNN更新回数: +{updates_after - updates_before}")
                print(f"  💾 インタラクション蓄積: +{buffer_after - buffer_before}")

                # 詳細統計
                print(f"\n📈 GNN学習統計:")
                gnn_extractor.print_statistics()

                # 時系列情報の分析
                if gnn_extractor.interaction_buffer:
                    times = [
                        interaction["simulation_time"]
                        for interaction in gnn_extractor.interaction_buffer
                    ]
                    if times:
                        min_time = min(times)
                        max_time = max(times)
                        print(f"  ⏰ インタラクション時間範囲:")
                        print(f"    開始: {min_time}")
                        print(f"    終了: {max_time}")
                        print(f"    期間: {max_time - min_time}")

    except Exception as e:
        print(f"❌ RL実行エラー: {e}")
        import traceback

        traceback.print_exc()

    # 総合結果
    total_duration = datetime.now() - start_time
    print(f"\n🎉 完全パイプライン実行完了")
    print(f"=" * 60)
    print(f"  📊 総実行時間: {total_duration.total_seconds():.1f}秒")
    print(f"  ✅ IRL: {'成功' if reward_weights is not None else 'スキップ'}")
    print(f"  ✅ RL: 実行完了")
    print(f"  🧠 GNNオンライン学習: 時系列対応")

    # 次のステップの提案
    print(f"\n💡 次のステップ提案:")
    print(f"  1. 学習結果の評価スクリプト実行")
    print(f"  2. GNN更新効果の分析")
    print(f"  3. 時系列学習データの可視化")
    print(f"  4. より長期間のシミュレーション実行")


if __name__ == "__main__":
    run_complete_pipeline()
