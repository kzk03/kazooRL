#!/usr/bin/env python3
"""
Kazoo 統合学習パイプライン

このスクリプトは以下を順次実行します：
1. 協力ネットワーク対応GNNモデルの訓練
2. 逆強化学習（IRL）による報酬重み学習
3. 強化学習（RL）による最終的なエージェント訓練

Usage:
    python scripts/full_training_pipeline.py [OPTIONS]

Options:
    --config PATH    設定ファイルのパス (default: configs/base.yaml)
    --skip-gnn       GNN訓練をスキップ
    --skip-irl       IRL訓練をスキップ
    --skip-rl        RL訓練をスキップ
    --production     プロダクション設定を使用
    --quiet          詳細ログを無効化
    --help           ヘルプ表示

Examples:
    python scripts/full_training_pipeline.py                    # フル実行
    python scripts/full_training_pipeline.py --production       # プロダクション設定
    python scripts/full_training_pipeline.py --skip-gnn         # GNNスキップ
    python scripts/full_training_pipeline.py --skip-gnn --skip-irl  # RLのみ
"""
import argparse
import json
import pickle
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from omegaconf import OmegaConf

# Kazooモジュールをインポート
sys.path.append(str(Path(__file__).resolve().parents[1]))
from kazoo.envs.oss_simple import OSSSimpleEnv
from kazoo.envs.task import Task
from kazoo.features.feature_extractor import FeatureExtractor


class FullTrainingPipeline:
    """統合学習パイプラインの実行クラス"""

    def __init__(self, config_path="configs/base.yaml", quiet=False):
        self.config_path = config_path
        self.cfg = OmegaConf.load(config_path)
        self.start_time = datetime.now()
        self.quiet = quiet

        # ログディレクトリ作成
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # ログファイル設定
        self.log_file = (
            log_dir / f"kazoo_training_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log"
        )

        # 初期メッセージ
        self.log(f"🚀 Kazoo統合学習パイプライン開始: {self.start_time}")
        self.log(f"📝 設定: {config_path}")
        self.log(f"📋 ログ: {self.log_file}")

    def log(self, message):
        """ログメッセージの出力"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"

        if not self.quiet:
            print(log_message)

        # ログファイルにも記録
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")

    def check_prerequisites(self):
        """前提条件の確認"""
        self.log("🔍 前提条件の確認中...")

        required_files = [
            self.cfg.env.backlog_path,
            self.cfg.env.dev_profiles_path,
            self.cfg.irl.expert_path,
        ]

        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            self.log(f"❌ 必要なファイルが見つかりません: {missing_files}")
            return False

        # データディレクトリの確認
        Path("data").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("outputs").mkdir(exist_ok=True)

        self.log("✅ 前提条件チェック完了")
        return True

    def train_collaborative_gnn(self):
        """協力ネットワーク対応GNNモデルの訓練"""
        self.log("🧠 協力ネットワーク対応GNN訓練開始...")

        try:
            # 協力ネットワーク構築（必要に応じて）
            if not Path("data/developer_collaboration_network.pt").exists():
                self.log("🔗 開発者協力ネットワークを構築中...")
                result = subprocess.run(
                    [
                        sys.executable,
                        "tools/data_processing/build_developer_network.py",
                    ],
                    capture_output=True,
                    text=True,
                    cwd=Path.cwd(),
                )

                if result.returncode != 0:
                    self.log(f"❌ 開発者ネットワーク構築エラー: {result.stderr}")
                    return False

                self.log("✅ 開発者協力ネットワーク構築完了")

            # GNNモデル訓練
            self.log("🏋️ GNNモデル訓練実行中...")
            start_time = time.time()

            result = subprocess.run(
                [sys.executable, "scripts/train_collaborative_gat.py"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )

            duration = time.time() - start_time

            if result.returncode != 0:
                self.log(f"❌ GNN訓練エラー: {result.stderr}")
                return False

            self.log(f"✅ GNN訓練完了 (所要時間: {duration:.1f}秒)")

            # 生成されたモデルファイルの確認
            required_models = [
                "data/gnn_model_collaborative.pt",
                "data/graph_collaborative.pt",
            ]

            for model_path in required_models:
                if not Path(model_path).exists():
                    self.log(
                        f"❌ 生成されるべきモデルファイルが見つかりません: {model_path}"
                    )
                    return False

            return True

        except Exception as e:
            self.log(f"❌ GNN訓練中の予期しないエラー: {e}")
            self.log(traceback.format_exc())
            return False

    def train_irl(self):
        """逆強化学習（IRL）による報酬重み学習"""
        self.log("🎯 逆強化学習（IRL）開始...")

        try:
            start_time = time.time()

            # IRLの実行
            self.log(
                f"📊 IRL設定: エポック数={self.cfg.irl.epochs}, 学習率={self.cfg.irl.learning_rate}"
            )

            # train_irl.pyの内容を直接実行
            self._run_irl_training()

            duration = time.time() - start_time
            self.log(f"✅ IRL訓練完了 (所要時間: {duration:.1f}秒)")

            # 学習済み重みの確認
            if not Path(self.cfg.irl.output_weights_path).exists():
                self.log(
                    f"❌ 学習済み重みファイルが生成されませんでした: {self.cfg.irl.output_weights_path}"
                )
                return False

            # 重みの分析
            weights = np.load(self.cfg.irl.output_weights_path)
            self.log(
                f"📈 学習済み重み統計: 平均={weights.mean():.4f}, 標準偏差={weights.std():.4f}"
            )
            self.log(f"📈 重み範囲: 最小={weights.min():.4f}, 最大={weights.max():.4f}")

            return True

        except Exception as e:
            self.log(f"❌ IRL訓練中のエラー: {e}")
            self.log(traceback.format_exc())
            return False

    def _run_irl_training(self):
        """IRL訓練の実際の実行"""
        self.log("📚 エキスパート軌跡とデータの読み込み...")

        # データ読み込み
        try:
            with open(self.cfg.irl.expert_path, "rb") as f:
                trajectories = pickle.load(f)
                expert_trajectory_steps = trajectories[0] if trajectories else []
        except Exception as e:
            raise Exception(f"エキスパート軌跡の読み込みエラー: {e}")

        if not expert_trajectory_steps:
            raise Exception("エキスパート軌跡にステップが含まれていません")

        with open(self.cfg.env.backlog_path, "r", encoding="utf-8") as f:
            backlog_data = json.load(f)
        with open(self.cfg.env.dev_profiles_path, "r", encoding="utf-8") as f:
            dev_profiles_data = yaml.safe_load(f)

        # 環境と特徴量抽出器の初期化
        env = OSSSimpleEnv(self.cfg, backlog_data, dev_profiles_data)
        feature_extractor = FeatureExtractor(self.cfg)
        feature_dim = len(feature_extractor.feature_names)

        all_tasks_db = {task.id: task for task in env.backlog}

        self.log(f"🔧 IRLモデル設定: 特徴量次元={feature_dim}")

        # モデル初期化
        reward_weights = torch.randn(feature_dim, requires_grad=True)
        optimizer = optim.Adam([reward_weights], lr=self.cfg.irl.learning_rate)

        self.log(
            f"🔄 訓練ループ開始: {len(expert_trajectory_steps)} エキスパートステップ"
        )

        # 訓練ループ
        for epoch in range(self.cfg.irl.epochs):
            total_loss = 0
            valid_steps = 0

            for step_data in expert_trajectory_steps:
                try:
                    optimizer.zero_grad()

                    # 軌跡データから状態と行動を取得
                    state = step_data["state"]
                    action_details = step_data["action_details"]

                    developer_id = action_details.get("developer")
                    expert_task_id = action_details.get("task_id")
                    event_timestamp = datetime.fromisoformat(
                        action_details.get("timestamp").replace("Z", "+00:00")
                    )

                    # データ有効性確認
                    developer_profile = dev_profiles_data.get(developer_id)
                    expert_task = all_tasks_db.get(expert_task_id)
                    if not developer_profile or not expert_task:
                        continue

                    developer_obj = {"name": developer_id, "profile": developer_profile}
                    env.current_time = event_timestamp

                    # 特徴量計算
                    expert_features = feature_extractor.get_features(
                        expert_task, developer_obj, env
                    )
                    expert_features = torch.from_numpy(expert_features).float()

                    # 他の可能な行動の特徴量
                    other_features_list = []
                    for other_task_id in state["open_task_ids"]:
                        if other_task_id != expert_task_id:
                            other_task = all_tasks_db.get(other_task_id)
                            if other_task:
                                features = feature_extractor.get_features(
                                    other_task, developer_obj, env
                                )
                                other_features_list.append(
                                    torch.from_numpy(features).float()
                                )

                    if not other_features_list:
                        continue

                    # 損失計算
                    expert_reward = torch.dot(reward_weights, expert_features)
                    other_rewards = torch.stack(
                        [torch.dot(reward_weights, f) for f in other_features_list]
                    )
                    log_sum_exp_other_rewards = torch.logsumexp(other_rewards, dim=0)

                    loss = -(expert_reward - log_sum_exp_other_rewards)
                    total_loss += loss.item()
                    valid_steps += 1

                    loss.backward()
                    optimizer.step()

                except Exception as e:
                    self.log(f"⚠️ ステップ処理エラー (スキップ): {e}")
                    continue

            if valid_steps > 0:
                avg_loss = total_loss / valid_steps
                if (epoch + 1) % 100 == 0 or epoch == 0:
                    self.log(
                        f"📈 エポック {epoch + 1}/{self.cfg.irl.epochs}, 平均損失: {avg_loss:.6f}, 有効ステップ: {valid_steps}"
                    )

        # 重みの保存
        self.log("💾 学習済み報酬重みの保存...")
        np.save(self.cfg.irl.output_weights_path, reward_weights.detach().numpy())

    def train_rl(self):
        """強化学習（RL）による最終エージェント訓練"""
        self.log("🤖 強化学習（RL）開始...")

        try:
            start_time = time.time()

            self.log(
                f"🎮 RL設定: タイムステップ={self.cfg.rl.total_timesteps}, 学習率={self.cfg.rl.learning_rate}"
            )

            # RLの実行
            result = subprocess.run(
                [sys.executable, "scripts/train_oss.py"],
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )

            duration = time.time() - start_time

            if result.returncode != 0:
                self.log(f"❌ RL訓練エラー: {result.stderr}")
                return False

            self.log(f"✅ RL訓練完了 (所要時間: {duration:.1f}秒)")

            # 学習済みモデルの確認
            model_paths = ["models/ppo_agent.pt"]
            for model_path in model_paths:
                if Path(model_path).exists():
                    self.log(f"✅ 学習済みモデル確認: {model_path}")
                else:
                    self.log(f"⚠️ 学習済みモデルが見つかりません: {model_path}")

            return True

        except Exception as e:
            self.log(f"❌ RL訓練中のエラー: {e}")
            self.log(traceback.format_exc())
            return False

    def generate_summary_report(self):
        """最終的なサマリーレポートの生成"""
        self.log("📊 サマリーレポート生成中...")

        end_time = datetime.now()
        total_duration = end_time - self.start_time

        report = {
            "kazoo_training_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration_seconds": total_duration.total_seconds(),
                "config_used": self.config_path,
            },
            "generated_files": {
                "gnn_model": "data/gnn_model_collaborative.pt",
                "gnn_graph": "data/graph_collaborative.pt",
                "irl_weights": self.cfg.irl.output_weights_path,
                "rl_model": "models/ppo_agent.pt",
                "collaboration_network": "data/developer_collaboration_network.pt",
            },
            "file_status": {},
        }

        # ファイル存在確認
        for name, path in report["generated_files"].items():
            exists = Path(path).exists()
            size = Path(path).stat().st_size if exists else 0
            report["file_status"][name] = {
                "exists": exists,
                "path": path,
                "size_bytes": size,
            }

        # レポート保存
        report_path = (
            self.log_file.parent
            / f"kazoo_report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.log(f"📋 レポート保存: {report_path}")
        self.log(f"⏱️ 総実行時間: {total_duration}")

        return report_path

    def run_full_pipeline(self, skip_gnn=False, skip_irl=False, skip_rl=False):
        """統合パイプラインの実行"""
        try:
            self.log("=" * 60)
            self.log("🚀 Kazoo統合学習パイプライン実行開始")
            self.log("=" * 60)

            # 前提条件チェック
            if not self.check_prerequisites():
                self.log("❌ 前提条件チェック失敗。実行を中止します。")
                return False

            success_steps = []

            # ステップ1: GNN訓練
            if not skip_gnn:
                self.log("\n" + "=" * 40)
                self.log("ステップ1: 協力ネットワーク対応GNN訓練")
                self.log("=" * 40)
                if self.train_collaborative_gnn():
                    success_steps.append("GNN")
                else:
                    self.log("❌ GNN訓練に失敗しました。")
                    return False
            else:
                self.log("⏭️ GNN訓練をスキップ")
                success_steps.append("GNN (スキップ)")

            # ステップ2: IRL訓練
            if not skip_irl:
                self.log("\n" + "=" * 40)
                self.log("ステップ2: 逆強化学習（IRL）")
                self.log("=" * 40)
                if self.train_irl():
                    success_steps.append("IRL")
                else:
                    self.log("❌ IRL訓練に失敗しました。")
                    return False
            else:
                self.log("⏭️ IRL訓練をスキップ")
                success_steps.append("IRL (スキップ)")

            # ステップ3: RL訓練
            if not skip_rl:
                self.log("\n" + "=" * 40)
                self.log("ステップ3: 強化学習（RL）")
                self.log("=" * 40)
                if self.train_rl():
                    success_steps.append("RL")
                else:
                    self.log("❌ RL訓練に失敗しました。")
                    return False
            else:
                self.log("⏭️ RL訓練をスキップ")
                success_steps.append("RL (スキップ)")

            # 最終レポート生成
            self.log("\n" + "=" * 40)
            self.log("ステップ4: サマリーレポート生成")
            self.log("=" * 40)
            report_path = self.generate_summary_report()

            self.log("\n" + "=" * 60)
            self.log("🎉 Kazoo統合学習パイプライン完了!")
            self.log(f"✅ 成功したステップ: {', '.join(success_steps)}")
            self.log(f"📋 詳細レポート: {report_path}")
            self.log("=" * 60)

            return True

        except Exception as e:
            self.log(f"❌ パイプライン実行中の予期しないエラー: {e}")
            self.log(traceback.format_exc())
            return False


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Kazoo統合学習パイプライン",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python scripts/full_training_pipeline.py                    # フル実行
  python scripts/full_training_pipeline.py --production       # プロダクション設定
  python scripts/full_training_pipeline.py --skip-gnn         # GNNスキップ
  python scripts/full_training_pipeline.py --skip-gnn --skip-irl  # RLのみ
  python scripts/full_training_pipeline.py --quiet            # 静粛モード

実行時間の目安:
  GNN訓練: 30分〜1時間
  IRL学習: 2〜4時間  
  RL学習: 8〜12時間
  合計: 10〜17時間（フル実行）
        """,
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="設定ファイルのパス (default: configs/base.yaml)",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="プロダクション設定を使用 (configs/production.yaml)",
    )
    parser.add_argument("--skip-gnn", action="store_true", help="GNN訓練をスキップ")
    parser.add_argument("--skip-irl", action="store_true", help="IRL訓練をスキップ")
    parser.add_argument("--skip-rl", action="store_true", help="RL訓練をスキップ")
    parser.add_argument("--quiet", action="store_true", help="詳細ログを無効化")

    args = parser.parse_args()

    # プロダクション設定の処理
    config_path = "configs/production.yaml" if args.production else args.config

    # パイプライン実行
    pipeline = FullTrainingPipeline(config_path, quiet=args.quiet)

    if not args.quiet:
        print(f"\n🚀 Kazoo統合学習パイプライン")
        print(f"📝 設定: {config_path}")
        skip_list = []
        if args.skip_gnn:
            skip_list.append("GNN")
        if args.skip_irl:
            skip_list.append("IRL")
        if args.skip_rl:
            skip_list.append("RL")
        if skip_list:
            print(f"⏭️ スキップ: {', '.join(skip_list)}")
        else:
            print(f"🔄 実行: 全ステップ（GNN → IRL → RL）")
        print()

    success = pipeline.run_full_pipeline(
        skip_gnn=args.skip_gnn, skip_irl=args.skip_irl, skip_rl=args.skip_rl
    )

    if not args.quiet:
        if success:
            print(f"\n✅ 処理が正常に完了しました！")
            print(f"📋 ログ: {pipeline.log_file}")
        else:
            print(f"\n❌ 処理中にエラーが発生しました")
            print(f"📋 ログ: {pipeline.log_file}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
