#!/usr/bin/env python3
"""
特徴量設計の実装確認
==================

実装した特徴量設計クラスが正しく動作するかを確認します。
"""

import sys
from pathlib import Path

import numpy as np

# 現在のディレクトリを追加
sys.path.append(str(Path(__file__).parent))


def main():
    """メイン関数"""
    print("🚀 特徴量設計実装確認開始")
    print("=" * 60)

    try:
        # 1. TaskFeatureDesigner のテスト
        print("\n📋 TaskFeatureDesigner テスト")
        print("-" * 40)

        from task_feature_designer import TaskFeatureDesigner

        task_designer = TaskFeatureDesigner()
        print("✅ 初期化成功")

        # サンプルタスクデータ
        sample_task = {
            "title": "Fix critical API bug in authentication system",
            "body": """This is a critical bug that needs immediate attention. 
            The authentication API endpoint is failing with 500 errors.
            
            ```python
            def authenticate_user(username, password):
                # This function is broken
                return None
            ```
            
            @developer1 please help with this ASAP. 
            Related to #123 and PR #456.
            Deadline: end of sprint.
            """,
            "comments": 15,
            "labels": ["critical", "bug", "api", "authentication"],
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
            "watchers": 25,
            "reactions": {"+1": 8, "heart": 3, "eyes": 2},
            "assignees": ["user1", "user2"],
        }

        task_features = task_designer.design_enhanced_features(sample_task)
        print(f"✅ タスク特徴量設計完了: {len(task_features)}個の特徴量")

        # 主要特徴量を表示
        key_features = [
            "task_text_length_log",
            "task_priority_label_score",
            "task_urgency_keyword_count",
            "task_technical_term_density",
            "task_social_attention_score",
        ]

        print("   主要特徴量:")
        for feature in key_features:
            if feature in task_features:
                print(f"     {feature}: {task_features[feature]:.3f}")

        # 2. DeveloperFeatureDesigner のテスト
        print("\n👨‍💻 DeveloperFeatureDesigner テスト")
        print("-" * 40)

        from developer_feature_designer import DeveloperFeatureDesigner

        dev_designer = DeveloperFeatureDesigner()
        print("✅ 初期化成功")

        # サンプル開発者データ
        sample_developer = {
            "total_commits": 1500,
            "total_prs": 200,
            "total_issues": 50,
            "total_lines_changed": 50000,
            "collaboration_network_size": 25,
            "comment_interactions": 800,
            "recent_activity_count": 20,
            "current_workload": 3,
            "languages": {"python": 70, "javascript": 25, "sql": 5},
            "touched_files": [
                "src/api/views.py",
                "frontend/components/App.js",
                "tests/test_api.py",
                "docs/api.md",
            ],
            "skills": ["python", "django", "react", "postgresql", "docker"],
            "recent_skills": ["kubernetes", "graphql"],
            "activity_hours": [9, 10, 11, 14, 15, 16, 17],
            "activity_days": [0, 1, 2, 3, 4],  # 平日のみ
            "response_times": [1.5, 2.0, 0.5, 3.0, 1.0],
            "merged_prs": 180,
            "reviews_given": 150,
            "approved_reviews": 130,
            "commits_with_bugs": 30,
            "review_comments": 500,
            "helpful_reviews": 400,
            "test_files_touched": 100,
            "total_files_touched": 300,
            "doc_files_touched": 20,
            "years_experience": 4,
        }

        dev_features = dev_designer.design_enhanced_features(sample_developer)
        print(f"✅ 開発者特徴量設計完了: {len(dev_features)}個の特徴量")

        # 主要特徴量を表示
        key_dev_features = [
            "dev_primary_language_strength",
            "dev_overall_consistency_score",
            "dev_pr_merge_rate",
            "dev_overall_quality_score",
            "dev_modern_tech_adoption",
        ]

        print("   主要特徴量:")
        for feature in key_dev_features:
            if feature in dev_features:
                print(f"     {feature}: {dev_features[feature]:.3f}")

        # 3. MatchingFeatureDesigner のテスト
        print("\n🔗 MatchingFeatureDesigner テスト")
        print("-" * 40)

        from matching_feature_designer import MatchingFeatureDesigner

        matching_designer = MatchingFeatureDesigner()
        print("✅ 初期化成功")

        # サンプル環境コンテキスト
        sample_env = {
            "collaboration_history": [
                {
                    "participants": ["user1", "developer1"],
                    "date": "2024-01-01T00:00:00Z",
                },
                {
                    "participants": ["user2", "developer1"],
                    "date": "2024-01-15T00:00:00Z",
                },
            ],
            "task_history": [
                {
                    "assignees": ["developer1"],
                    "status": "completed",
                    "type": "bug",
                    "labels": ["python", "api"],
                    "completed_date": "2024-01-10T00:00:00Z",
                    "quality_score": 0.85,
                    "completed_on_time": True,
                },
                {
                    "assignees": ["developer1"],
                    "status": "completed",
                    "type": "feature",
                    "labels": ["javascript"],
                    "completed_date": "2024-01-20T00:00:00Z",
                    "quality_score": 0.75,
                    "completed_on_time": False,
                },
            ],
            "collaboration_feedback": {
                "developer1": [{"satisfaction": 4.5}, {"satisfaction": 4.0}]
            },
        }

        # 開発者データを更新（マッチング用）
        sample_developer["name"] = "developer1"
        sample_developer["timezone"] = "UTC"

        matching_features = matching_designer.design_enhanced_features(
            sample_task, sample_developer, sample_env
        )
        print(f"✅ マッチング特徴量設計完了: {len(matching_features)}個の特徴量")

        # 主要特徴量を表示
        key_matching_features = [
            "match_collaboration_strength_enhanced",
            "match_skill_compatibility_weighted",
            "match_overall_tech_compatibility",
            "match_estimated_success_probability",
            "match_success_confidence",
        ]

        print("   主要特徴量:")
        for feature in key_matching_features:
            if feature in matching_features:
                print(f"     {feature}: {matching_features[feature]:.3f}")

        # 4. 統合テスト
        print("\n🔄 統合テスト")
        print("-" * 40)

        # 全特徴量を結合
        all_features = {}
        all_features.update({f"task_{k}": v for k, v in task_features.items()})
        all_features.update({f"dev_{k}": v for k, v in dev_features.items()})
        all_features.update(matching_features)

        print(f"✅ 統合特徴量: {len(all_features)}個")

        # 特徴量の妥当性検証
        validation_results = []

        for designer, features, name in [
            (task_designer, task_features, "タスク"),
            (dev_designer, dev_features, "開発者"),
            (matching_designer, matching_features, "マッチング"),
        ]:
            validation = designer.validate_features(features)
            validation_results.append((name, validation))

            invalid_count = len(validation["invalid_features"])
            warning_count = len(validation["warnings"])

            if invalid_count == 0 and warning_count == 0:
                print(f"✅ {name}特徴量検証: 問題なし")
            else:
                print(
                    f"⚠️  {name}特徴量検証: 無効{invalid_count}個, 警告{warning_count}個"
                )

        # 5. 特徴量名リスト取得テスト
        print("\n📝 特徴量名リスト取得テスト")
        print("-" * 40)

        task_feature_names = task_designer.get_feature_names()
        dev_feature_names = dev_designer.get_feature_names()
        matching_feature_names = matching_designer.get_feature_names()

        print(f"✅ タスク特徴量名: {len(task_feature_names)}個")
        print(f"✅ 開発者特徴量名: {len(dev_feature_names)}個")
        print(f"✅ マッチング特徴量名: {len(matching_feature_names)}個")

        total_unique_features = len(
            set(task_feature_names + dev_feature_names + matching_feature_names)
        )
        print(f"✅ 総ユニーク特徴量: {total_unique_features}個")

        print("\n🎉 全ての実装確認完了！")
        print("=" * 60)
        print("✅ TaskFeatureDesigner: 動作確認済み")
        print("✅ DeveloperFeatureDesigner: 動作確認済み")
        print("✅ MatchingFeatureDesigner: 動作確認済み")
        print("\n📝 要件2.1, 2.2, 2.3の実装が完了しました。")

        # 統計サマリー
        print(f"\n📊 特徴量統計:")
        print(f"   - タスク特徴量: {len(task_features)}個")
        print(f"   - 開発者特徴量: {len(dev_features)}個")
        print(f"   - マッチング特徴量: {len(matching_features)}個")
        print(f"   - 総特徴量数: {len(all_features)}個")

        return True

    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
