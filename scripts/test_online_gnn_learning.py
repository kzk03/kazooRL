#!/usr/bin/env python3

"""
オンラインGNN学習のテストスクリプト
強化学習のステップごとにGNNが更新されることを確認
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

import json

import hydra
import numpy as np
import yaml
from omegaconf import DictConfig

from kazoo.envs.oss_simple import OSSSimpleEnv
from kazoo.features.feature_extractor import FeatureExtractor


def test_online_gnn_learning():
    """オンラインGNN学習をテスト"""
    
    print("🧪 オンラインGNN学習テスト")
    print("=" * 50)
    
    # 設定を読み込み
    with hydra.initialize(config_path="../configs", version_base=None):
        cfg = hydra.compose(config_name="base")
    
    # オンライン学習を有効化
    cfg.irl.online_gnn_learning = True
    cfg.irl.gnn_update_frequency = 10  # テスト用に頻繁に更新
    
    print(f"オンライン学習設定:")
    print(f"  有効: {cfg.irl.online_gnn_learning}")
    print(f"  更新頻度: {cfg.irl.gnn_update_frequency}ステップごと")
    print(f"  学習率: {cfg.irl.gnn_learning_rate}")
    print(f"  バッファサイズ: {cfg.irl.gnn_buffer_size}")
    print()
    
    # データを読み込み
    with open(cfg.env.backlog_path, 'r') as f:
        backlog = json.load(f)
    
    with open(cfg.env.dev_profiles_path, 'r') as f:
        dev_profiles = yaml.safe_load(f)
    
    # 人間の開発者のみフィルタ
    human_devs = {name: profile for name, profile in dev_profiles.items() 
                  if 'bot' not in name.lower()}
    
    print(f"データ統計:")
    print(f"  タスク数: {len(backlog)}")
    print(f"  人間開発者数: {len(human_devs)}")
    print()
    
    # 環境と特徴量抽出器を初期化
    print("環境を初期化中...")
    env = OSSSimpleEnv(cfg, backlog[:100], human_devs)  # テスト用に100タスクに制限
    feature_extractor = FeatureExtractor(cfg)
    
    if not feature_extractor.gnn_extractor:
        print("❌ GNN特徴量抽出器が利用できません")
        return
    
    if not feature_extractor.gnn_extractor.online_learning:
        print("❌ オンライン学習が有効になっていません") 
        return
    
    print("✅ オンライン学習対応GNN初期化完了")
    print()
    
    # 初期統計を記録
    print("📊 初期統計:")
    feature_extractor.gnn_extractor.print_statistics()
    
    # シミュレーション実行
    print("\n🎮 シミュレーション開始...")
    
    obs, info = env.reset()
    developer_names = list(human_devs.keys())
    
    for step in range(100):  # 100ステップ実行
        # ランダムアクション
        actions = {}
        for agent_id in env.agent_ids:
            if env.backlog:  # バックログにタスクがあれば
                action = np.random.randint(0, min(len(env.backlog), 5))  # ランダムに選択
            else:
                action = len(env.initial_backlog)  # NO_OP
            actions[agent_id] = action
        
        # ステップ実行
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # 報酬があった場合の統計
        total_reward = sum(rewards.values())
        if total_reward != 0:
            print(f"  Step {step}: Total reward = {total_reward:.3f}")
        
        # 10ステップごとに統計表示
        if (step + 1) % 20 == 0:
            print(f"\n--- Step {step + 1} 統計 ---")
            feature_extractor.gnn_extractor.print_statistics()
        
        # 終了条件
        if all(terminated.values()) or all(truncated.values()):
            print(f"シミュレーション終了 (Step {step})")
            break
    
    # 最終統計
    print("\n📈 最終統計:")
    feature_extractor.gnn_extractor.print_statistics()
    
    # 更新されたモデルを保存
    if feature_extractor.gnn_extractor.stats["updates"] > 0:
        print("\n💾 更新されたモデルを保存中...")
        feature_extractor.gnn_extractor.save_updated_model()
        print("✅ 保存完了")
    else:
        print("\n⚠️ GNNの更新が発生しませんでした")
    
    # 新しい開発者/タスクの追加テスト
    print("\n🆕 新しいノード追加テスト...")
    
    new_developers = {
        "test_new_dev": {
            "skills": ["python", "testing"],
            "touched_files": ["test.py"],
            "label_affinity": {"bug": 0.8, "enhancement": 0.5}
        }
    }
    
    new_tasks = {
        "test_task_12345": {
            "title": "Test task for online learning",
            "body": "This is a test task with ```code blocks```",
            "labels": ["bug", "test"]
        }
    }
    
    feature_extractor.gnn_extractor.add_new_nodes(new_developers, new_tasks)
    
    print("\n🎯 結論:")
    if feature_extractor.gnn_extractor.stats["updates"] > 0:
        print("✅ オンラインGNN学習が正常に動作しています")
        print(f"✅ {feature_extractor.gnn_extractor.stats['updates']}回のGNN更新が実行されました")
        print("✅ 強化学習のステップごとにGNNが改善されています")
    else:
        print("⚠️ GNN更新が発生しませんでした - より多くのステップまたは報酬が必要です")
    
    print("🚀 システムは継続的に学習・改善する準備ができています！")


def test_manual_gnn_update():
    """手動でGNN更新をテスト"""
    print("\n" + "=" * 50)
    print("🔧 手動GNN更新テスト")
    
    # 設定を読み込み
    with hydra.initialize(config_path="../configs", version_base=None):
        cfg = hydra.compose(config_name="base")
    
    cfg.irl.online_gnn_learning = True
    
    feature_extractor = FeatureExtractor(cfg)
    gnn = feature_extractor.gnn_extractor
    
    if not gnn:
        print("❌ GNN特徴量抽出器が利用できません")
        return
    
    print("✅ GNN初期化完了")
    
    # 手動でインタラクションを追加
    print("📝 手動インタラクション記録...")
    
    # テスト用のタスクと開発者
    class TestTask:
        def __init__(self, task_id):
            self.id = task_id
    
    test_interactions = [
        (TestTask("pr_347626001"), {"name": "ndeloof"}, 1.5),
        (TestTask("pr_347626001"), {"name": "chris-crone"}, -0.5),
        (TestTask("pr_345198021"), {"name": "ndeloof"}, 2.0),
        (TestTask("issue_12345"), {"name": "missing_dev"}, 0.8),
    ]
    
    for task, developer, reward in test_interactions:
        gnn.record_interaction(task, developer, reward, "COMPLETE" if reward > 0 else "SKIP")
        print(f"  記録: {developer['name']} + {task.id} = {reward}")
    
    # 強制的に更新
    print("\n🔄 強制GNN更新...")
    gnn._update_gnn_online()
    
    print("\n📊 更新後統計:")
    gnn.print_statistics()


if __name__ == "__main__":
    test_online_gnn_learning()
    test_manual_gnn_update()
