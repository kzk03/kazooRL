#!/usr/bin/env python3
"""
Kazoo統合学習パイプライン - 1から実行
グラフ生成 → GAT → 逆強化学習 → 強化学習の順で実行
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd, description):
    """コマンドを実行し、結果を表示"""
    print(f"\n🚀 {description}")
    print(f"実行コマンド: {cmd}")
    print("=" * 60)
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} 完了")
    else:
        print(f"❌ {description} 失敗 (exit code: {result.returncode})")
        return False
    return True

def main():
    """メイン実行関数"""
    start_time = datetime.now()
    print(f"🎯 Kazoo 1から学習パイプライン開始: {start_time}")
    
    # 作業ディレクトリを確認
    project_root = Path(__file__).parent
    print(f"📁 作業ディレクトリ: {project_root.absolute()}")
    
    # 1. データ準備の確認
    print("\n📊 データファイル確認中...")
    required_files = [
        "data/backlog_training.json",
        "data/expert_trajectories.pkl", 
        "data/labels.pt",
        "data/developer_collaboration_network.pt"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 見つかりません")
            return False
    
    # 2. グラフデータ生成（学習用: 2022年除外）
    if not run_command(
        "python tools/data_processing/generate_graph.py --exclude-years 2022 --output-suffix _training",
        "グラフデータ生成（学習用: 2022年除外）"
    ):
        return False
    
    # 3. GAT訓練（協力ネットワーク対応）
    if not run_command(
        "python scripts/train_collaborative_gat.py",
        "GAT（Graph Attention Network）訓練"
    ):
        return False
    
    # 4. 逆強化学習（IRL）
    if not run_command(
        "python scripts/train_irl.py",
        "逆強化学習（Inverse Reinforcement Learning）"
    ):
        return False
    
    # 5. 強化学習（RL）
    if not run_command(
        "python scripts/train_oss.py",
        "強化学習（Reinforcement Learning）"
    ):
        return False
    
    # 6. 評価
    if not run_command(
        "python scripts/eval_oss.py",
        "学習済みモデルの評価"
    ):
        print("⚠️ 評価は失敗しましたが、学習は完了しています")
    
    end_time = datetime.now()
    elapsed = end_time - start_time
    
    print("\n" + "=" * 60)
    print(f"🎉 Kazoo 1から学習パイプライン完了!")
    print(f"⏱️ 実行時間: {elapsed}")
    print(f"📅 完了時刻: {end_time}")
    
    print("\n📂 生成されたファイル:")
    generated_files = [
        "data/graph_training.pt",
        "data/gnn_model_collaborative.pt",
        "data/graph_collaborative.pt", 
        "data/learned_weights_training.npy",
        "models/ppo_agent.pt"
    ]
    
    for file_path in generated_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 生成されませんでした")

if __name__ == "__main__":
    main()
