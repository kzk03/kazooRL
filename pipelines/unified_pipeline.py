#!/usr/bin/env python3
"""
Kazoo 統合学習パイプライン - 最終版
GAT → IRL → RL の完全な学習フローを実行
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class KazooUnifiedPipeline:
    """統合されたKazoo学習パイプライン"""

    def __init__(self):
        self.start_time = datetime.now()
        self.project_root = Path(__file__).parent.parent
        self.log_file = (
            self.project_root
            / "outputs"
            / f"pipeline_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log"
        )
        self.log_file.parent.mkdir(exist_ok=True)

    def log(self, message):
        """ログ出力"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")

    def run_command(self, cmd, description, working_dir=None):
        """コマンドを実行"""
        self.log(f"\n🚀 {description}")
        self.log(f"実行コマンド: {cmd}")
        self.log("=" * 60)

        if working_dir:
            original_dir = Path.cwd()
            os.chdir(working_dir)

        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.stdout:
                self.log(f"出力:\n{result.stdout}")
            if result.stderr:
                self.log(f"エラー:\n{result.stderr}")

            if result.returncode == 0:
                self.log(f"✅ {description} 完了")
                return True
            else:
                self.log(f"❌ {description} 失敗 (exit code: {result.returncode})")
                return False
        finally:
            if working_dir:
                os.chdir(original_dir)

    def check_prerequisites(self):
        """前提条件をチェック"""
        self.log("📋 前提条件チェック中...")

        # 必要なディレクトリの存在確認
        required_dirs = ["data", "configs", "outputs", "models"]
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                dir_path.mkdir(exist_ok=True)
                self.log(f"ディレクトリ作成: {dir_name}")

        # 基本的な設定ファイルの確認
        config_files = ["configs/base.yaml", "configs/dev_profiles.yaml"]

        missing_files = []
        for config_file in config_files:
            if not (self.project_root / config_file).exists():
                missing_files.append(config_file)

        if missing_files:
            self.log(f"⚠️ 以下のファイルが見つかりません: {missing_files}")
            self.log("データ前処理を実行します...")
            return self.prepare_data()

        self.log("✅ 前提条件チェック完了")
        return True

    def prepare_data(self):
        """データ前処理を実行"""
        self.log("\n📊 データ前処理開始")

        data_commands = [
            ("python data_processing/generate_graph.py", "グラフ生成"),
            ("python data_processing/generate_profiles.py", "プロファイル生成"),
            ("python data_processing/generate_labels.py", "ラベル生成"),
            ("python data_processing/generate_backlog.py", "バックログ生成"),
        ]

        for cmd, desc in data_commands:
            if not self.run_command(cmd, desc, self.project_root):
                self.log(f"❌ データ前処理失敗: {desc}")
                return False

        self.log("✅ データ前処理完了")
        return True

    def train_gat(self):
        """GATトレーニング"""
        self.log("\n🧠 GAT (Graph Attention Network) トレーニング開始")

        # まず協力関係グラフの構築
        if not self.run_command(
            "python training/gat/train_collaborative_gat.py",
            "協力関係GATトレーニング",
            self.project_root,
        ):
            # フォールバック: 通常のGNNトレーニング
            self.log(
                "協力関係GATトレーニングに失敗、通常のGNNトレーニングにフォールバック"
            )
            return self.run_command(
                "python training/gat/train_gnn.py", "GNNトレーニング", self.project_root
            )

        self.log("✅ GATトレーニング完了")
        return True

    def train_irl(self):
        """IRLトレーニング"""
        self.log("\n🎯 IRL (Inverse Reinforcement Learning) トレーニング開始")

        return self.run_command(
            "python training/irl/train_irl.py", "IRLトレーニング", self.project_root
        )

    def train_rl(self):
        """RLトレーニング"""
        self.log("\n🎮 RL (Reinforcement Learning) トレーニング開始")

        return self.run_command(
            "python training/rl/train_oss.py", "RLトレーニング", self.project_root
        )

    def run_evaluation(self):
        """評価実行"""
        self.log("\n📊 モデル評価開始")

        eval_commands = [
            ("python evaluation/evaluate_models.py", "モデル評価"),
            ("python evaluation/test_features.py", "特徴量テスト"),
        ]

        success_count = 0
        for cmd, desc in eval_commands:
            if self.run_command(cmd, desc, self.project_root):
                success_count += 1

        self.log(f"評価完了: {success_count}/{len(eval_commands)} 成功")
        return success_count > 0

    def generate_reports(self):
        """分析レポート生成"""
        self.log("\n📈 分析レポート生成開始")

        report_commands = [
            ("python analysis/reports/summary_report.py", "総合レポート生成"),
            ("python analysis/reports/irl_analysis.py", "IRL分析レポート"),
            ("python analysis/reports/gat_analysis.py", "GAT分析レポート"),
            ("python analysis/visualization/plot_results.py", "可視化生成"),
        ]

        for cmd, desc in report_commands:
            self.run_command(cmd, desc, self.project_root)

        self.log("✅ 分析レポート生成完了")
        return True

    def run_full_pipeline(self):
        """完全なパイプラインを実行"""
        self.log("🚀 Kazoo統合学習パイプライン開始")
        self.log(f"実行時刻: {self.start_time}")
        self.log(f"ログファイル: {self.log_file}")

        # ステップ1: 前提条件チェック
        if not self.check_prerequisites():
            self.log("❌ 前提条件チェック失敗")
            return False

        # ステップ2: GATトレーニング
        if not self.train_gat():
            self.log("❌ GATトレーニング失敗")
            return False

        # ステップ3: IRLトレーニング
        if not self.train_irl():
            self.log("❌ IRLトレーニング失敗")
            return False

        # ステップ4: RLトレーニング
        if not self.train_rl():
            self.log("❌ RLトレーニング失敗")
            return False

        # ステップ5: 評価
        self.run_evaluation()

        # ステップ6: レポート生成
        self.generate_reports()

        # 完了
        end_time = datetime.now()
        duration = end_time - self.start_time
        self.log(f"\n🎉 パイプライン完了!")
        self.log(f"実行時間: {duration}")
        self.log(f"終了時刻: {end_time}")

        return True


def run_gat_only():
    """GAT単体実行"""
    pipeline = KazooUnifiedPipeline()
    pipeline.log("🧠 GAT単体トレーニング開始")

    if pipeline.check_prerequisites():
        success = pipeline.train_gat()
        if success:
            pipeline.log("✅ GAT単体トレーニング完了")
        return success
    return False


def run_irl_only():
    """IRL単体実行"""
    pipeline = KazooUnifiedPipeline()
    pipeline.log("🎯 IRL単体トレーニング開始")

    if pipeline.check_prerequisites():
        success = pipeline.train_irl()
        if success:
            pipeline.log("✅ IRL単体トレーニング完了")
        return success
    return False


def run_rl_only():
    """RL単体実行"""
    pipeline = KazooUnifiedPipeline()
    pipeline.log("🎮 RL単体トレーニング開始")

    if pipeline.check_prerequisites():
        success = pipeline.train_rl()
        if success:
            pipeline.log("✅ RL単体トレーニング完了")
        return success
    return False


def main():
    """メイン実行"""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == "gat":
            run_gat_only()
        elif mode == "irl":
            run_irl_only()
        elif mode == "rl":
            run_rl_only()
        elif mode == "full":
            pipeline = KazooUnifiedPipeline()
            pipeline.run_full_pipeline()
        else:
            print("使用法: python unified_pipeline.py [gat|irl|rl|full]")
            print("  gat  - GAT単体実行")
            print("  irl  - IRL単体実行")
            print("  rl   - RL単体実行")
            print("  full - 完全パイプライン実行")
    else:
        # デフォルトは完全パイプライン
        pipeline = KazooUnifiedPipeline()
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
