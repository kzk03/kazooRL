#!/usr/bin/env python3
"""
時系列GNNオンライン学習のテストスクリプト

このスクリプトは、シミュレーション時間の進行に伴ってGNNが
段階的に学習・更新されることを確認します。
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

# プロジェクトのルートをパスに追加
sys.path.append(str(Path(__file__).resolve().parents[1]))

from kazoo.envs.oss_simple import OSSSimpleEnv
from kazoo.features.feature_extractor import FeatureExtractor
from kazoo.learners.independent_ppo_controller import IndependentPPOController


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """時系列GNNオンライン学習テスト"""
    
    print("=" * 70)
    print("🕒 時系列GNNオンライン学習テスト")
    print("=" * 70)
    
    # 設定確認
    print("\n📋 設定確認:")
    print(f"  - GNN使用: {cfg.irl.get('use_gnn', False)}")
    print(f"  - オンライン学習: {cfg.irl.get('online_gnn_learning', False)}")
    print(f"  - 更新頻度: {cfg.irl.get('gnn_update_frequency', 'N/A')} ステップごと")
    print(f"  - 時間窓: {cfg.irl.get('gnn_time_window_hours', 'N/A')} 時間")
    print(f"  - RL総ステップ数: {cfg.rl.get('total_timesteps', 'N/A')}")
    
    # データ読み込み
    print("\n📚 データ読み込み中...")
    with open(cfg.env.backlog_path, "r", encoding="utf-8") as f:
        backlog = json.load(f)
    with open(cfg.env.dev_profiles_path, "r", encoding="utf-8") as f:
        dev_profiles = yaml.safe_load(f)
    
    print(f"  ✅ バックログ: {len(backlog)} タスク")
    print(f"  ✅ 開発者プロファイル: {len(dev_profiles)} 人")
    
    # 環境初期化
    print("\n🌍 環境初期化中...")
    env = OSSSimpleEnv(
        config=cfg,
        backlog=backlog,
        dev_profiles=dev_profiles,
        reward_weights_path=cfg.irl.output_weights_path,
    )
    
    # GNN特徴量抽出器の状態確認
    print("\n🔍 GNN特徴量抽出器の状態確認:")
    if hasattr(env, 'feature_extractor') and hasattr(env.feature_extractor, 'gnn_extractor'):
        gnn_extractor = env.feature_extractor.gnn_extractor
        if gnn_extractor:
            print(f"  ✅ GNN特徴量抽出器: 利用可能")
            print(f"  ✅ オンライン学習: {'有効' if gnn_extractor.online_learning else '無効'}")
            print(f"  ✅ 開発者ノード数: {len(gnn_extractor.dev_id_to_idx)}")
            print(f"  ✅ タスクノード数: {len(gnn_extractor.task_id_to_idx)}")
            print(f"  ✅ 時間窓: {gnn_extractor.time_window_hours} 時間")
            
            # 初期シミュレーション時間を記録
            initial_time = env.current_time
            print(f"  📅 初期シミュレーション時間: {initial_time}")
            
            # 手動でいくつかのインタラクションを記録して時系列動作をテスト
            print("\n🧪 手動インタラクション記録テスト:")
            
            # 異なる時間でのインタラクション
            test_interactions = [
                (initial_time, "positive", 1.0),
                (initial_time + timedelta(hours=2), "assignment", 0.5),
                (initial_time + timedelta(hours=8), "positive", 0.8),
                (initial_time + timedelta(hours=12), "negative", -0.3),
                (initial_time + timedelta(hours=20), "positive", 1.2),
                (initial_time + timedelta(hours=26), "assignment", 0.6),  # 時間窓外
            ]
            
            # テスト用の開発者とタスクを選択
            dev_names = list(dev_profiles.keys())[:3]
            task_ids = [task['id'] for task in backlog[:3]]
            
            for i, (sim_time, action_type, reward) in enumerate(test_interactions):
                dev_name = dev_names[i % len(dev_names)]
                task_id = task_ids[i % len(task_ids)]
                
                # 簡単なオブジェクトを作成
                class MockTask:
                    def __init__(self, tid):
                        self.id = tid
                
                class MockDeveloper:
                    def __init__(self, name):
                        self.name = name
                    def get(self, key, default=None):
                        if key == "name":
                            return self.name
                        return default
                
                task = MockTask(task_id)
                developer = MockDeveloper(dev_name)
                
                print(f"  📝 記録 {i+1}: {sim_time} - {dev_name} + {task_id} = {reward} ({action_type})")
                gnn_extractor.record_interaction(
                    task, developer, reward, action_type, simulation_time=sim_time
                )
            
            # バッファの状態確認
            buffer_size = len(gnn_extractor.interaction_buffer)
            print(f"\n📊 インタラクションバッファ: {buffer_size} 件")
            
            if buffer_size > 0:
                # 時間範囲確認
                times = [interaction['simulation_time'] for interaction in gnn_extractor.interaction_buffer]
                min_time = min(times)
                max_time = max(times)
                print(f"  📅 時間範囲: {min_time} ～ {max_time}")
                print(f"  ⏱️  期間: {max_time - min_time}")
                
                # 手動でGNN更新をトリガー
                print("\n🔄 手動GNN更新実行:")
                gnn_extractor._update_gnn_online()
                
                # 統計表示
                print("\n📈 最終統計:")
                gnn_extractor.print_statistics()
            
        else:
            print("  ❌ GNN特徴量抽出器: 利用不可")
    else:
        print("  ❌ 特徴量抽出器が見つかりません")
    
    # PPOコントローラー初期化・実行
    print("\n🤖 強化学習での時系列GNN動作確認:")
    controller = IndependentPPOController(env=env, config=cfg)
    
    try:
        print("  🚀 強化学習開始...")
        controller.learn(total_timesteps=cfg.rl.total_timesteps)
        
        # 最終的なGNN状態確認
        if hasattr(env, 'feature_extractor') and hasattr(env.feature_extractor, 'gnn_extractor'):
            gnn_extractor = env.feature_extractor.gnn_extractor
            if gnn_extractor:
                print("\n📊 強化学習後のGNN統計:")
                gnn_extractor.print_statistics()
                
                # 時系列分析
                if gnn_extractor.interaction_buffer:
                    times = [interaction['simulation_time'] for interaction in gnn_extractor.interaction_buffer]
                    min_time = min(times)
                    max_time = max(times)
                    print(f"\n⏰ 時系列分析:")
                    print(f"  - 最初のインタラクション: {min_time}")
                    print(f"  - 最後のインタラクション: {max_time}")
                    print(f"  - 総期間: {max_time - min_time}")
                    print(f"  - 現在のシミュレーション時間: {env.current_time}")
                    
                    # 時間窓内の件数
                    latest_time = max_time
                    cutoff_time = latest_time - timedelta(hours=gnn_extractor.time_window_hours)
                    recent_count = sum(1 for interaction in gnn_extractor.interaction_buffer 
                                     if interaction['simulation_time'] >= cutoff_time)
                    print(f"  - 時間窓内のインタラクション: {recent_count}/{len(gnn_extractor.interaction_buffer)}")
                
                # モデル保存
                if gnn_extractor.stats["updates"] > 0:
                    gnn_extractor.save_updated_model("data/gnn_model_timeseries_updated.pt")
                    print("💾 時系列学習後のGNNモデルを保存しました")
        
        print("\n✅ 時系列GNN学習テスト完了!")
        
    except Exception as e:
        print(f"\n❌ テスト中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
