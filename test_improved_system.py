#!/usr/bin/env python3
"""
改良システムの簡単なテスト
エラーの特定と基本動作確認
"""

import json
import sys
from pathlib import Path

import yaml

# パッケージのパスを追加
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """インポートテスト"""
    print("=== インポートテスト ===")
    
    try:
        from kazoo.envs.oss_simple import OSSSimpleEnv
        print("✅ OSSSimpleEnv インポート成功")
    except Exception as e:
        print(f"❌ OSSSimpleEnv インポートエラー: {e}")
        return False
    
    try:
        from kazoo.envs.improved_reward_system import ImprovedRewardSystem
        print("✅ ImprovedRewardSystem インポート成功")
    except Exception as e:
        print(f"⚠️ ImprovedRewardSystem インポートエラー: {e}")
    
    try:
        from kazoo.envs.action_space_reducer import HierarchicalActionSpace
        print("✅ HierarchicalActionSpace インポート成功")
    except Exception as e:
        print(f"⚠️ HierarchicalActionSpace インポートエラー: {e}")
    
    try:
        from kazoo.envs.observation_processor import ObservationProcessor
        print("✅ ObservationProcessor インポート成功")
    except Exception as e:
        print(f"⚠️ ObservationProcessor インポートエラー: {e}")
    
    return True

def test_config_loading():
    """設定ファイル読み込みテスト"""
    print("\n=== 設定ファイルテスト ===")
    
    config_path = "configs/simple_test.yaml"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ 設定ファイル読み込み成功: {config_path}")
        return config
    except Exception as e:
        print(f"❌ 設定ファイル読み込みエラー: {e}")
        return None

def test_data_loading():
    """データファイル読み込みテスト"""
    print("\n=== データファイルテスト ===")
    
    # サンプルデータの作成
    sample_backlog = [
        {
            "id": 1,
            "title": "Test Task 1",
            "body": "This is a test task",
            "labels": [{"name": "bug"}],
            "created_at": "2022-01-01T00:00:00Z",
            "updated_at": "2022-01-01T00:00:00Z",
            "comments_count": 5,
            "assignees": []
        },
        {
            "id": 2,
            "title": "Test Task 2",
            "body": "Another test task",
            "labels": [{"name": "enhancement"}],
            "created_at": "2022-01-02T00:00:00Z",
            "updated_at": "2022-01-02T00:00:00Z",
            "comments_count": 2,
            "assignees": []
        }
    ]
    
    sample_dev_profiles = {
        "dev1": {
            "rank": 1000,
            "total_commits": 500,
            "python_commits": 200,
            "javascript_commits": 100,
            "bug_fixes": 50,
            "doc_commits": 20
        },
        "dev2": {
            "rank": 2000,
            "total_commits": 300,
            "python_commits": 150,
            "javascript_commits": 80,
            "bug_fixes": 30,
            "doc_commits": 10
        }
    }
    
    print("✅ サンプルデータ作成成功")
    return sample_backlog, sample_dev_profiles

def test_environment_creation():
    """環境作成テスト"""
    print("\n=== 環境作成テスト ===")
    
    try:
        from kazoo.envs.oss_simple import OSSSimpleEnv

        # 設定とデータの準備
        config = test_config_loading()
        if not config:
            return False
        
        backlog, dev_profiles = test_data_loading()
        
        # 設定をオブジェクト形式に変換
        class SimpleConfig:
            def __init__(self, config_dict):
                for key, value in config_dict.items():
                    if isinstance(value, dict):
                        setattr(self, key, SimpleConfig(value))
                    else:
                        setattr(self, key, value)
            
            def get(self, key, default=None):
                return getattr(self, key, default)
        
        config_obj = SimpleConfig(config)
        
        # 環境の作成
        env = OSSSimpleEnv(
            config=config_obj,
            backlog=backlog,
            dev_profiles=dev_profiles
        )
        
        print(f"✅ 環境作成成功")
        print(f"   - エージェント数: {len(env.agent_ids)}")
        print(f"   - タスク数: {len(env.backlog)}")
        print(f"   - 行動空間: {env.action_space}")
        print(f"   - 観測空間: {env.observation_space}")
        
        return env
        
    except Exception as e:
        print(f"❌ 環境作成エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_environment_step():
    """環境ステップテスト"""
    print("\n=== 環境ステップテスト ===")
    
    env = test_environment_creation()
    if not env:
        return False
    
    try:
        # 環境のリセット
        obs, info = env.reset()
        print("✅ 環境リセット成功")
        
        # ランダム行動でステップ実行
        actions = {}
        for agent_id in env.agent_ids:
            actions[agent_id] = env.action_space[agent_id].sample()
        
        next_obs, rewards, terminateds, truncateds, infos = env.step(actions)
        
        print("✅ 環境ステップ実行成功")
        print(f"   - 報酬: {rewards}")
        print(f"   - 終了フラグ: {terminateds}")
        
        return True
        
    except Exception as e:
        print(f"❌ 環境ステップエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メインテスト関数"""
    print("🧪 改良システムテスト開始")
    print("=" * 50)
    
    # 各テストの実行
    tests = [
        ("インポート", test_imports),
        ("設定ファイル", test_config_loading),
        ("データ読み込み", test_data_loading),
        ("環境作成", test_environment_creation),
        ("環境ステップ", test_environment_step)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result is not False and result is not None:
                results.append((test_name, "✅ 成功"))
            else:
                results.append((test_name, "❌ 失敗"))
        except Exception as e:
            results.append((test_name, f"❌ エラー: {e}"))
    
    # 結果サマリー
    print("\n" + "=" * 50)
    print("🏁 テスト結果サマリー")
    print("=" * 50)
    
    for test_name, result in results:
        print(f"{test_name:15s}: {result}")
    
    success_count = sum(1 for _, result in results if "✅" in result)
    total_count = len(results)
    
    print(f"\n成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        print("🎉 全テスト成功！")
    else:
        print("⚠️ 一部テストが失敗しました。上記のエラーを確認してください。")

if __name__ == "__main__":
    main()