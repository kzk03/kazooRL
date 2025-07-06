#!/usr/bin/env python3
"""
Kazoo 統合学習パイプライン
GAT → IRL → RL の完全な学習フローを実行
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

class KazooPipeline:
    """Kazoo学習パイプラインクラス"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.project_root = Path(__file__).parent.parent
        
    def run_command(self, cmd, description, working_dir=None):
        """コマンドを実行"""
        print(f"\n🚀 {description}")
        print(f"実行コマンド: {cmd}")
        print("=" * 60)
        
        if working_dir:
            original_dir = Path.cwd()
            os.chdir(working_dir)
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
            
            if result.returncode == 0:
                print(f"✅ {description} 完了")
                return True
            else:
                print(f"❌ {description} 失敗 (exit code: {result.returncode})")
                return False
        finally:
            if working_dir:
                os.chdir(original_dir)
    
    def check_prerequisites(self):
        """前提条件をチェック"""
        print("📋 前提条件チェック中...")
        
        required_files = [
            "data/backlog_training.json",
            "data/expert_trajectories.pkl",
            "data/labels.pt",
            "configs/base_training.yaml"
        ]
        
        all_exist = True
        for file_path in required_files:
            if (self.project_root / file_path).exists():
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path} - 見つかりません")
                all_exist = False
        
        return all_exist
    
    def run_gat_training(self):
        """GAT訓練を実行"""
        return self.run_command(
            "python training/gat/train_gat.py",
            "GAT (Graph Attention Network) 訓練",
            self.project_root
        )
    
    def run_irl_training(self):
        """IRL訓練を実行"""
        return self.run_command(
            "python training/irl/train_irl.py", 
            "IRL (Inverse Reinforcement Learning) 訓練",
            self.project_root
        )
    
    def run_rl_training(self):
        """RL訓練を実行"""
        return self.run_command(
            "python training/rl/train_rl.py",
            "RL (Reinforcement Learning) 訓練", 
            self.project_root
        )
    
    def run_analysis(self):
        """結果分析を実行"""
        return self.run_command(
            "python analysis/reports/irl_analysis.py",
            "結果分析・レポート生成",
            self.project_root
        )
    
    def run_evaluation(self):
        """評価を実行"""
        return self.run_command(
            "python evaluation/evaluate_models.py",
            "モデル評価",
            self.project_root
        )
    
    def run_full_pipeline(self):
        """完全パイプラインを実行"""
        print(f"🎯 Kazoo統合学習パイプライン開始")
        print(f"📅 開始時刻: {self.start_time}")
        print(f"📁 作業ディレクトリ: {self.project_root}")
        print("=" * 80)
        
        # 前提条件チェック
        if not self.check_prerequisites():
            print("❌ 前提条件が満たされていません")
            return False
        
        steps = [
            ("GAT訓練", self.run_gat_training),
            ("IRL訓練", self.run_irl_training), 
            ("RL訓練", self.run_rl_training),
            ("結果分析", self.run_analysis),
            ("モデル評価", self.run_evaluation)
        ]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            if not step_func():
                failed_steps.append(step_name)
                print(f"⚠️ {step_name}で失敗しましたが、継続します")
        
        # 完了レポート
        end_time = datetime.now()
        elapsed = end_time - self.start_time
        
        print("\n" + "=" * 80)
        print(f"🎉 Kazoo統合学習パイプライン完了!")
        print(f"⏱️ 実行時間: {elapsed}")
        print(f"📅 完了時刻: {end_time}")
        
        if failed_steps:
            print(f"\n⚠️ 失敗したステップ: {', '.join(failed_steps)}")
        else:
            print(f"\n✅ 全ステップが正常に完了しました!")
        
        # 生成されたファイルの確認
        self.check_generated_files()
        
        return len(failed_steps) == 0
    
    def check_generated_files(self):
        """生成されたファイルを確認"""
        print(f"\n📂 生成ファイル確認:")
        
        expected_files = [
            "data/gnn_model_collaborative.pt",
            "data/graph_collaborative.pt",
            "data/learned_weights_training.npy",
            "models/ppo_agent.pt"
        ]
        
        for file_path in expected_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                size = full_path.stat().st_size
                print(f"✅ {file_path} ({size:,} bytes)")
            else:
                print(f"❌ {file_path} - 生成されませんでした")

def main():
    """メイン実行"""
    import os
    
    pipeline = KazooPipeline()
    success = pipeline.run_full_pipeline()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
