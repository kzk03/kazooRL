"""
マッチング特徴量設計器
====================

既存マッチング特徴量の改良と新規特徴量の追加機能を提供します。
時間的近接性、技術的適合性、成功履歴の特徴量を実装します。
"""

import logging
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class MatchingFeatureDesigner:
    """マッチング特徴量設計器

    既存マッチング特徴量の改良（協力強度、スキル適合性、ファイル専門性関連度）を実装。
    時間的近接性、技術的適合性、成功履歴の新規特徴量を追加。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 設定辞書（技術スタック、重み設定など）
        """
        self.config = config or {}

        # 技術スタック階層
        self.tech_stack_hierarchy = self.config.get(
            "tech_stack_hierarchy",
            {
                "languages": ["python", "javascript", "java", "c++", "go", "rust"],
                "frameworks": ["django", "flask", "react", "vue", "spring", "express"],
                "databases": [
                    "postgresql",
                    "mysql",
                    "mongodb",
                    "redis",
                    "elasticsearch",
                ],
                "tools": ["docker", "kubernetes", "git", "jenkins", "terraform"],
                "cloud": ["aws", "gcp", "azure", "heroku"],
            },
        )

        # 協力強度の重み
        self.collaboration_weights = self.config.get(
            "collaboration_weights",
            {
                "direct_collaboration": 1.0,
                "indirect_collaboration": 0.5,
                "review_interaction": 0.7,
                "issue_interaction": 0.3,
            },
        )

        # 成功指標の重み
        self.success_weights = self.config.get(
            "success_weights",
            {
                "completion_rate": 0.4,
                "quality_score": 0.3,
                "timeliness": 0.2,
                "satisfaction": 0.1,
            },
        )

        logger.info(f"MatchingFeatureDesigner初期化完了")

    def design_enhanced_features(
        self,
        task_data: Dict[str, Any],
        developer_data: Dict[str, Any],
        env_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """強化されたマッチング特徴量を設計

        Args:
            task_data: タスクデータ辞書
            developer_data: 開発者データ辞書
            env_context: 環境コンテキスト（履歴データ、他の開発者情報など）

        Returns:
            強化された特徴量辞書
        """
        features = {}

        # 既存特徴量の改良
        features.update(
            self._improve_existing_features(task_data, developer_data, env_context)
        )

        # 時間的近接性特徴量
        features.update(
            self._extract_temporal_proximity_features(
                task_data, developer_data, env_context
            )
        )

        # 技術的適合性特徴量
        features.update(
            self._extract_technical_compatibility_features(task_data, developer_data)
        )

        # 成功履歴特徴量
        features.update(
            self._extract_success_history_features(
                task_data, developer_data, env_context
            )
        )

        logger.debug(f"マッチング特徴量設計完了: {len(features)}個の特徴量")
        return features

    def _improve_existing_features(
        self,
        task_data: Dict[str, Any],
        developer_data: Dict[str, Any],
        env_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """既存特徴量の改良

        Args:
            task_data: タスクデータ
            developer_data: 開発者データ
            env_context: 環境コンテキスト

        Returns:
            改良された既存特徴量
        """
        features = {}

        # 1. 協力強度の改良
        collaboration_history = (
            env_context.get("collaboration_history", []) if env_context else []
        )
        developer_name = developer_data.get("name", "")
        task_author = task_data.get("author", "")

        # 直接協力の強度計算
        direct_collaborations = [
            collab
            for collab in collaboration_history
            if developer_name in collab.get("participants", [])
            and task_author in collab.get("participants", [])
        ]

        if direct_collaborations:
            # 協力の頻度と最近性を考慮
            recent_collaborations = [
                collab
                for collab in direct_collaborations
                if self._is_recent(collab.get("date"), days=90)
            ]

            collaboration_strength = (
                len(recent_collaborations) * 2.0 + len(direct_collaborations) * 1.0
            )
            features["match_collaboration_strength_enhanced"] = float(
                np.log1p(collaboration_strength)
            )
        else:
            features["match_collaboration_strength_enhanced"] = 0.0

        # 2. スキル適合性の改良
        task_required_skills = self._extract_required_skills(task_data)
        developer_skills = set(developer_data.get("skills", []))

        if task_required_skills and developer_skills:
            # 完全一致スキル
            exact_matches = task_required_skills.intersection(developer_skills)
            features["match_skill_exact_matches"] = float(len(exact_matches))

            # 部分一致スキル（類似技術）
            partial_matches = self._calculate_skill_similarity(
                task_required_skills, developer_skills
            )
            features["match_skill_partial_matches"] = float(partial_matches)

            # スキル適合度（重み付き）
            skill_compatibility = self._calculate_weighted_skill_compatibility(
                task_required_skills, developer_skills
            )
            features["match_skill_compatibility_weighted"] = float(skill_compatibility)
        else:
            features["match_skill_exact_matches"] = 0.0
            features["match_skill_partial_matches"] = 0.0
            features["match_skill_compatibility_weighted"] = 0.0

        # 3. ファイル専門性関連度の改良
        task_files = set(task_data.get("changed_files", []))
        developer_files = set(developer_data.get("touched_files", []))

        if task_files and developer_files:
            # 直接的なファイル経験
            direct_file_matches = task_files.intersection(developer_files)
            features["match_direct_file_experience"] = float(len(direct_file_matches))

            # ディレクトリレベルの経験
            directory_matches = self._calculate_directory_overlap(
                task_files, developer_files
            )
            features["match_directory_experience"] = float(directory_matches)

            # ファイルタイプの経験
            file_type_matches = self._calculate_file_type_overlap(
                task_files, developer_files
            )
            features["match_file_type_experience"] = float(file_type_matches)

            # 総合ファイル関連度
            total_file_relevance = (
                len(direct_file_matches) * 3.0
                + directory_matches * 2.0
                + file_type_matches * 1.0
            )
            features["match_file_relevance_total"] = float(total_file_relevance)
        else:
            features["match_direct_file_experience"] = 0.0
            features["match_directory_experience"] = 0.0
            features["match_file_type_experience"] = 0.0
            features["match_file_relevance_total"] = 0.0

        return features

    def _extract_temporal_proximity_features(
        self,
        task_data: Dict[str, Any],
        developer_data: Dict[str, Any],
        env_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """時間的近接性特徴量の抽出

        Args:
            task_data: タスクデータ
            developer_data: 開発者データ
            env_context: 環境コンテキスト

        Returns:
            時間的近接性特徴量辞書
        """
        features = {}

        developer_name = developer_data.get("name", "")

        # 1. 最近の協力日数
        collaboration_history = (
            env_context.get("collaboration_history", []) if env_context else []
        )
        recent_collaborations = [
            collab
            for collab in collaboration_history
            if developer_name in collab.get("participants", [])
        ]

        if recent_collaborations:
            # 最も最近の協力からの日数
            latest_collaboration = max(
                recent_collaborations, key=lambda x: x.get("date", "")
            )
            days_since_last_collaboration = self._calculate_days_since(
                latest_collaboration.get("date")
            )
            features["match_days_since_last_collaboration"] = float(
                days_since_last_collaboration
            )
            features["match_days_since_last_collaboration_log"] = float(
                np.log1p(days_since_last_collaboration)
            )

            # 最近30日間の協力回数
            recent_30d_collaborations = [
                collab
                for collab in recent_collaborations
                if self._is_recent(collab.get("date"), days=30)
            ]
            features["match_recent_collaboration_count"] = float(
                len(recent_30d_collaborations)
            )
        else:
            features["match_days_since_last_collaboration"] = 365.0  # デフォルト1年
            features["match_days_since_last_collaboration_log"] = float(np.log1p(365.0))
            features["match_recent_collaboration_count"] = 0.0

        # 2. 活動時間重複
        task_activity_hours = task_data.get("activity_hours", [])
        developer_activity_hours = developer_data.get("activity_hours", [])

        if task_activity_hours and developer_activity_hours:
            # 活動時間の重複度
            task_hours_set = set(task_activity_hours)
            dev_hours_set = set(developer_activity_hours)

            overlap_hours = task_hours_set.intersection(dev_hours_set)
            union_hours = task_hours_set.union(dev_hours_set)

            if union_hours:
                features["match_activity_time_overlap"] = float(
                    len(overlap_hours) / len(union_hours)
                )
            else:
                features["match_activity_time_overlap"] = 0.0

            # ピーク活動時間の近さ
            task_peak_hour = (
                Counter(task_activity_hours).most_common(1)[0][0]
                if task_activity_hours
                else 12
            )
            dev_peak_hour = (
                Counter(developer_activity_hours).most_common(1)[0][0]
                if developer_activity_hours
                else 12
            )

            hour_distance = min(
                abs(task_peak_hour - dev_peak_hour),
                24 - abs(task_peak_hour - dev_peak_hour),
            )
            features["match_peak_hour_distance"] = float(hour_distance)
            features["match_peak_hour_similarity"] = float(1.0 / (1.0 + hour_distance))
        else:
            features["match_activity_time_overlap"] = 0.5  # デフォルト
            features["match_peak_hour_distance"] = 6.0
            features["match_peak_hour_similarity"] = 0.14

        # 3. タイムゾーン適合性
        task_timezone = task_data.get("timezone", "UTC")
        developer_timezone = developer_data.get("timezone", "UTC")

        timezone_compatibility = self._calculate_timezone_compatibility(
            task_timezone, developer_timezone
        )
        features["match_timezone_compatibility"] = float(timezone_compatibility)

        # 4. 応答時間予測
        developer_avg_response_time = developer_data.get(
            "avg_response_time", 24.0
        )  # 時間
        task_urgency = task_data.get("urgency_score", 0.5)  # 0-1

        # 緊急度に基づく応答時間予測
        predicted_response_time = developer_avg_response_time * (2.0 - task_urgency)
        features["match_predicted_response_time"] = float(predicted_response_time)
        features["match_predicted_response_time_log"] = float(
            np.log1p(predicted_response_time)
        )

        # 応答時間適合性（短いほど良い）
        features["match_response_time_fitness"] = float(
            1.0 / (1.0 + predicted_response_time / 24.0)
        )

        return features

    def _extract_technical_compatibility_features(
        self, task_data: Dict[str, Any], developer_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """技術的適合性特徴量の抽出

        Args:
            task_data: タスクデータ
            developer_data: 開発者データ

        Returns:
            技術的適合性特徴量辞書
        """
        features = {}

        # 1. 技術スタック重複
        task_tech_stack = self._extract_tech_stack(task_data)
        developer_tech_stack = self._extract_developer_tech_stack(developer_data)

        for category in self.tech_stack_hierarchy.keys():
            task_techs = task_tech_stack.get(category, set())
            dev_techs = developer_tech_stack.get(category, set())

            if task_techs or dev_techs:
                overlap = task_techs.intersection(dev_techs)
                union = task_techs.union(dev_techs)

                if union:
                    overlap_ratio = len(overlap) / len(union)
                else:
                    overlap_ratio = 0.0

                features[f"match_tech_{category}_overlap"] = float(len(overlap))
                features[f"match_tech_{category}_overlap_ratio"] = float(overlap_ratio)
            else:
                features[f"match_tech_{category}_overlap"] = 0.0
                features[f"match_tech_{category}_overlap_ratio"] = 0.0

        # 2. 言語習熟度マッチング
        task_languages = task_tech_stack.get("languages", set())
        developer_languages = developer_data.get("languages", {})

        language_proficiency_score = 0.0
        for lang in task_languages:
            if lang in developer_languages:
                # 言語使用経験を習熟度として使用
                proficiency = developer_languages[lang] / 100.0  # 0-1に正規化
                language_proficiency_score += proficiency

        features["match_language_proficiency_score"] = float(language_proficiency_score)

        # 3. フレームワーク経験適合性
        task_frameworks = task_tech_stack.get("frameworks", set())
        developer_skills = set(developer_data.get("skills", []))

        framework_experience_score = 0.0
        for framework in task_frameworks:
            if framework in developer_skills:
                framework_experience_score += 1.0
            else:
                # 類似フレームワークの経験をチェック
                similar_frameworks = self._find_similar_frameworks(framework)
                for similar in similar_frameworks:
                    if similar in developer_skills:
                        framework_experience_score += 0.5
                        break

        features["match_framework_experience_score"] = float(framework_experience_score)

        # 4. アーキテクチャ親和性
        task_architecture_hints = self._extract_architecture_hints(task_data)
        developer_architecture_experience = (
            self._extract_developer_architecture_experience(developer_data)
        )

        architecture_compatibility = 0.0
        for arch_type in task_architecture_hints:
            if arch_type in developer_architecture_experience:
                architecture_compatibility += developer_architecture_experience[
                    arch_type
                ]

        features["match_architecture_compatibility"] = float(architecture_compatibility)

        # 5. 技術的複雑度適合性
        task_complexity = task_data.get("technical_complexity", 0.5)  # 0-1
        developer_experience_level = self._calculate_developer_experience_level(
            developer_data
        )

        # 複雑度と経験レベルのマッチング
        complexity_match = 1.0 - abs(task_complexity - developer_experience_level)
        features["match_complexity_experience_fit"] = float(complexity_match)

        # 6. 総合技術適合性スコア
        tech_compatibility_components = [
            features.get("match_language_proficiency_score", 0.0) * 0.3,
            features.get("match_framework_experience_score", 0.0) * 0.25,
            features.get("match_architecture_compatibility", 0.0) * 0.2,
            features.get("match_complexity_experience_fit", 0.0) * 0.25,
        ]

        features["match_overall_tech_compatibility"] = float(
            sum(tech_compatibility_components)
        )

        return features

    def _extract_success_history_features(
        self,
        task_data: Dict[str, Any],
        developer_data: Dict[str, Any],
        env_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """成功履歴特徴量の抽出

        Args:
            task_data: タスクデータ
            developer_data: 開発者データ
            env_context: 環境コンテキスト

        Returns:
            成功履歴特徴量辞書
        """
        features = {}

        developer_name = developer_data.get("name", "")
        task_history = env_context.get("task_history", []) if env_context else []

        # 開発者の過去のタスク履歴を取得
        developer_tasks = [
            task for task in task_history if developer_name in task.get("assignees", [])
        ]

        # 1. 過去成功率
        if developer_tasks:
            completed_tasks = [
                task for task in developer_tasks if task.get("status") == "completed"
            ]
            success_rate = len(completed_tasks) / len(developer_tasks)
            features["match_historical_success_rate"] = float(success_rate)

            # 最近の成功率（過去6ヶ月）
            recent_tasks = [
                task
                for task in developer_tasks
                if self._is_recent(task.get("completed_date"), days=180)
            ]

            if recent_tasks:
                recent_completed = [
                    task for task in recent_tasks if task.get("status") == "completed"
                ]
                recent_success_rate = len(recent_completed) / len(recent_tasks)
                features["match_recent_success_rate"] = float(recent_success_rate)
            else:
                features["match_recent_success_rate"] = features[
                    "match_historical_success_rate"
                ]
        else:
            features["match_historical_success_rate"] = 0.5  # デフォルト
            features["match_recent_success_rate"] = 0.5

        # 2. 類似タスク完了率
        current_task_type = task_data.get("type", "unknown")
        current_task_labels = set(task_data.get("labels", []))

        similar_tasks = []
        for task in developer_tasks:
            # タスクタイプまたはラベルが類似
            if (
                task.get("type") == current_task_type
                or len(set(task.get("labels", [])).intersection(current_task_labels))
                > 0
            ):
                similar_tasks.append(task)

        if similar_tasks:
            similar_completed = [
                task for task in similar_tasks if task.get("status") == "completed"
            ]
            similar_completion_rate = len(similar_completed) / len(similar_tasks)
            features["match_similar_task_completion_rate"] = float(
                similar_completion_rate
            )
        else:
            features["match_similar_task_completion_rate"] = features[
                "match_historical_success_rate"
            ]

        # 3. 協力満足度
        collaboration_feedback = (
            env_context.get("collaboration_feedback", {}) if env_context else {}
        )
        developer_feedback = collaboration_feedback.get(developer_name, [])

        if developer_feedback:
            avg_satisfaction = np.mean(
                [feedback.get("satisfaction", 3.0) for feedback in developer_feedback]
            )
            # 1-5スケールを0-1に正規化
            normalized_satisfaction = (avg_satisfaction - 1.0) / 4.0
            features["match_collaboration_satisfaction"] = float(
                normalized_satisfaction
            )
        else:
            features["match_collaboration_satisfaction"] = 0.5  # デフォルト

        # 4. 品質履歴
        if developer_tasks:
            # 平均品質スコア
            quality_scores = [
                task.get("quality_score", 0.5)
                for task in completed_tasks
                if task.get("quality_score")
            ]
            if quality_scores:
                avg_quality = np.mean(quality_scores)
                features["match_historical_quality_score"] = float(avg_quality)
            else:
                features["match_historical_quality_score"] = 0.5

            # 納期遵守率
            on_time_tasks = [
                task for task in completed_tasks if task.get("completed_on_time", True)
            ]
            if completed_tasks:
                timeliness_rate = len(on_time_tasks) / len(completed_tasks)
                features["match_timeliness_rate"] = float(timeliness_rate)
            else:
                features["match_timeliness_rate"] = 0.5
        else:
            features["match_historical_quality_score"] = 0.5
            features["match_timeliness_rate"] = 0.5

        # 5. 成功確率推定
        # 複数の要因を組み合わせて成功確率を推定
        success_factors = [
            features["match_recent_success_rate"]
            * self.success_weights["completion_rate"],
            features["match_historical_quality_score"]
            * self.success_weights["quality_score"],
            features["match_timeliness_rate"] * self.success_weights["timeliness"],
            features["match_collaboration_satisfaction"]
            * self.success_weights["satisfaction"],
        ]

        estimated_success_probability = sum(success_factors)
        features["match_estimated_success_probability"] = float(
            estimated_success_probability
        )

        # 6. リスク評価
        # 失敗リスクの要因を評価
        risk_factors = []

        # 作業負荷リスク
        current_workload = developer_data.get("current_workload", 0)
        workload_risk = min(current_workload / 10.0, 1.0)  # 10タスク以上で最大リスク
        risk_factors.append(workload_risk * 0.3)

        # 経験不足リスク
        tech_compatibility = features.get("match_overall_tech_compatibility", 0.5)
        experience_risk = 1.0 - tech_compatibility
        risk_factors.append(experience_risk * 0.4)

        # 時間的制約リスク
        response_time_fitness = features.get("match_response_time_fitness", 0.5)
        time_risk = 1.0 - response_time_fitness
        risk_factors.append(time_risk * 0.3)

        total_risk = sum(risk_factors)
        features["match_failure_risk_score"] = float(total_risk)
        features["match_success_confidence"] = float(1.0 - total_risk)

        return features

    # ヘルパーメソッド
    def _is_recent(self, date_str: Optional[str], days: int = 30) -> bool:
        """日付が最近かどうかを判定"""
        if not date_str:
            return False

        try:
            date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return (datetime.now() - date.replace(tzinfo=None)).days <= days
        except:
            return False

    def _calculate_days_since(self, date_str: Optional[str]) -> int:
        """指定日からの経過日数を計算"""
        if not date_str:
            return 365

        try:
            date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return (datetime.now() - date.replace(tzinfo=None)).days
        except:
            return 365

    def _extract_required_skills(self, task_data: Dict[str, Any]) -> Set[str]:
        """タスクから必要スキルを抽出"""
        skills = set()

        # ラベルからスキルを抽出
        labels = task_data.get("labels", [])
        for label in labels:
            if label.lower() in self.tech_stack_hierarchy.get("languages", []):
                skills.add(label.lower())

        # タイトル・本文からスキルを抽出
        text_content = (
            task_data.get("title", "") + " " + task_data.get("body", "")
        ).lower()

        for category, techs in self.tech_stack_hierarchy.items():
            for tech in techs:
                if tech in text_content:
                    skills.add(tech)

        return skills

    def _calculate_skill_similarity(
        self, required_skills: Set[str], developer_skills: Set[str]
    ) -> float:
        """スキルの類似度を計算"""
        similarity_score = 0.0

        for req_skill in required_skills:
            if req_skill in developer_skills:
                similarity_score += 1.0
            else:
                # 類似スキルをチェック
                similar_skills = self._find_similar_skills(req_skill)
                for similar in similar_skills:
                    if similar in developer_skills:
                        similarity_score += 0.5
                        break

        return similarity_score

    def _find_similar_skills(self, skill: str) -> List[str]:
        """類似スキルを検索"""
        similar_skills = []

        # 同じカテゴリ内の技術を類似とみなす
        for category, techs in self.tech_stack_hierarchy.items():
            if skill in techs:
                similar_skills.extend([tech for tech in techs if tech != skill])
                break

        return similar_skills

    def _calculate_weighted_skill_compatibility(
        self, required_skills: Set[str], developer_skills: Set[str]
    ) -> float:
        """重み付きスキル適合度を計算"""
        if not required_skills:
            return 0.0

        compatibility_score = 0.0
        total_weight = 0.0

        # カテゴリ別の重み
        category_weights = {
            "languages": 1.0,
            "frameworks": 0.8,
            "databases": 0.6,
            "tools": 0.4,
            "cloud": 0.5,
        }

        for skill in required_skills:
            weight = 0.5  # デフォルト重み

            # スキルのカテゴリを特定して重みを設定
            for category, techs in self.tech_stack_hierarchy.items():
                if skill in techs:
                    weight = category_weights.get(category, 0.5)
                    break

            total_weight += weight

            if skill in developer_skills:
                compatibility_score += weight

        return compatibility_score / total_weight if total_weight > 0 else 0.0

    def _calculate_directory_overlap(
        self, task_files: Set[str], developer_files: Set[str]
    ) -> float:
        """ディレクトリレベルの重複を計算"""
        task_dirs = set()
        dev_dirs = set()

        for file_path in task_files:
            dirs = file_path.split("/")[:-1]  # ファイル名を除く
            for i in range(len(dirs)):
                task_dirs.add("/".join(dirs[: i + 1]))

        for file_path in developer_files:
            dirs = file_path.split("/")[:-1]
            for i in range(len(dirs)):
                dev_dirs.add("/".join(dirs[: i + 1]))

        if not task_dirs or not dev_dirs:
            return 0.0

        overlap = task_dirs.intersection(dev_dirs)
        return len(overlap)

    def _calculate_file_type_overlap(
        self, task_files: Set[str], developer_files: Set[str]
    ) -> float:
        """ファイルタイプの重複を計算"""
        task_extensions = set()
        dev_extensions = set()

        for file_path in task_files:
            if "." in file_path:
                ext = file_path.split(".")[-1].lower()
                task_extensions.add(ext)

        for file_path in developer_files:
            if "." in file_path:
                ext = file_path.split(".")[-1].lower()
                dev_extensions.add(ext)

        if not task_extensions or not dev_extensions:
            return 0.0

        overlap = task_extensions.intersection(dev_extensions)
        return len(overlap)

    def _calculate_timezone_compatibility(self, tz1: str, tz2: str) -> float:
        """タイムゾーン適合性を計算"""
        # 簡単な実装：同じタイムゾーンなら1.0、異なれば距離に基づく
        if tz1 == tz2:
            return 1.0

        # タイムゾーン間の時差を推定（簡易版）
        timezone_offsets = {"UTC": 0, "EST": -5, "PST": -8, "JST": 9, "CET": 1}

        offset1 = timezone_offsets.get(tz1, 0)
        offset2 = timezone_offsets.get(tz2, 0)

        time_diff = abs(offset1 - offset2)
        # 時差が大きいほど適合性が低い
        compatibility = max(0.0, 1.0 - time_diff / 12.0)

        return compatibility

    def _extract_tech_stack(self, task_data: Dict[str, Any]) -> Dict[str, Set[str]]:
        """タスクから技術スタックを抽出"""
        tech_stack = {category: set() for category in self.tech_stack_hierarchy.keys()}

        text_content = (
            task_data.get("title", "") + " " + task_data.get("body", "")
        ).lower()

        for category, techs in self.tech_stack_hierarchy.items():
            for tech in techs:
                if tech in text_content:
                    tech_stack[category].add(tech)

        return tech_stack

    def _extract_developer_tech_stack(
        self, developer_data: Dict[str, Any]
    ) -> Dict[str, Set[str]]:
        """開発者から技術スタックを抽出"""
        tech_stack = {category: set() for category in self.tech_stack_hierarchy.keys()}

        skills = developer_data.get("skills", [])
        languages = developer_data.get("languages", {}).keys()

        all_techs = list(skills) + list(languages)

        for category, techs in self.tech_stack_hierarchy.items():
            for tech in techs:
                if tech in [t.lower() for t in all_techs]:
                    tech_stack[category].add(tech)

        return tech_stack

    def _find_similar_frameworks(self, framework: str) -> List[str]:
        """類似フレームワークを検索"""
        # 簡単な類似性マッピング
        similarity_map = {
            "django": ["flask", "fastapi"],
            "flask": ["django", "fastapi"],
            "react": ["vue", "angular"],
            "vue": ["react", "angular"],
            "angular": ["react", "vue"],
            "spring": ["express", "fastapi"],
            "express": ["spring", "flask"],
        }

        return similarity_map.get(framework, [])

    def _extract_architecture_hints(self, task_data: Dict[str, Any]) -> List[str]:
        """タスクからアーキテクチャのヒントを抽出"""
        architecture_keywords = {
            "microservices": ["microservice", "microservices", "service-oriented"],
            "monolith": ["monolith", "monolithic"],
            "serverless": ["serverless", "lambda", "function"],
            "api": ["api", "rest", "graphql"],
            "frontend": ["frontend", "ui", "client-side"],
            "backend": ["backend", "server-side", "api"],
        }

        text_content = (
            task_data.get("title", "") + " " + task_data.get("body", "")
        ).lower()

        detected_architectures = []
        for arch_type, keywords in architecture_keywords.items():
            if any(keyword in text_content for keyword in keywords):
                detected_architectures.append(arch_type)

        return detected_architectures

    def _extract_developer_architecture_experience(
        self, developer_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """開発者のアーキテクチャ経験を抽出"""
        # ファイルパスやスキルからアーキテクチャ経験を推定
        touched_files = developer_data.get("touched_files", [])
        skills = developer_data.get("skills", [])

        architecture_experience = {
            "microservices": 0.0,
            "monolith": 0.0,
            "serverless": 0.0,
            "api": 0.0,
            "frontend": 0.0,
            "backend": 0.0,
        }

        # ファイルパスから推定
        for file_path in touched_files:
            file_path_lower = file_path.lower()

            if any(term in file_path_lower for term in ["service", "microservice"]):
                architecture_experience["microservices"] += 0.1
            if any(
                term in file_path_lower for term in ["api", "controller", "endpoint"]
            ):
                architecture_experience["api"] += 0.1
            if any(
                term in file_path_lower
                for term in ["frontend", "ui", "component", "view"]
            ):
                architecture_experience["frontend"] += 0.1
            if any(
                term in file_path_lower for term in ["backend", "server", "service"]
            ):
                architecture_experience["backend"] += 0.1

        # スキルから推定
        skills_lower = [skill.lower() for skill in skills]

        if any(
            skill in skills_lower for skill in ["microservices", "docker", "kubernetes"]
        ):
            architecture_experience["microservices"] += 0.5
        if any(
            skill in skills_lower
            for skill in ["aws lambda", "serverless", "azure functions"]
        ):
            architecture_experience["serverless"] += 0.5
        if any(
            skill in skills_lower for skill in ["rest api", "graphql", "api design"]
        ):
            architecture_experience["api"] += 0.5

        return architecture_experience

    def _calculate_developer_experience_level(
        self, developer_data: Dict[str, Any]
    ) -> float:
        """開発者の経験レベルを計算（0-1）"""
        factors = []

        # 年数経験
        years_experience = developer_data.get("years_experience", 0)
        experience_score = min(years_experience / 10.0, 1.0)  # 10年で最大
        factors.append(experience_score * 0.3)

        # 総コミット数
        total_commits = developer_data.get("total_commits", 0)
        commit_score = min(
            np.log1p(total_commits) / np.log1p(10000), 1.0
        )  # 10000コミットで最大
        factors.append(commit_score * 0.2)

        # スキル数
        skill_count = len(developer_data.get("skills", []))
        skill_score = min(skill_count / 20.0, 1.0)  # 20スキルで最大
        factors.append(skill_score * 0.2)

        # 品質スコア
        quality_score = developer_data.get("overall_quality_score", 0.5)
        factors.append(quality_score * 0.3)

        return sum(factors)

    def get_feature_names(self) -> List[str]:
        """設計される特徴量名のリストを取得

        Returns:
            特徴量名のリスト
        """
        # サンプルデータで特徴量名を取得
        sample_task = {
            "title": "Fix API endpoint",
            "body": "Need to fix the Python Django API endpoint",
            "author": "user1",
            "changed_files": ["src/api/views.py", "tests/test_api.py"],
            "labels": ["bug", "python"],
            "type": "bug",
            "urgency_score": 0.7,
            "technical_complexity": 0.6,
            "activity_hours": [9, 10, 14, 15],
            "timezone": "UTC",
        }

        sample_developer = {
            "name": "developer1",
            "skills": ["python", "django", "react"],
            "languages": {"python": 80, "javascript": 60},
            "touched_files": ["src/api/views.py", "frontend/components/App.js"],
            "years_experience": 3,
            "total_commits": 500,
            "avg_response_time": 4.0,
            "activity_hours": [9, 10, 11, 14, 15, 16],
            "timezone": "UTC",
            "current_workload": 2,
            "overall_quality_score": 0.8,
        }

        sample_env = {
            "collaboration_history": [
                {
                    "participants": ["user1", "developer1"],
                    "date": "2024-01-01T00:00:00Z",
                }
            ],
            "task_history": [
                {
                    "assignees": ["developer1"],
                    "status": "completed",
                    "type": "bug",
                    "labels": ["python"],
                    "completed_date": "2024-01-15T00:00:00Z",
                    "quality_score": 0.8,
                    "completed_on_time": True,
                }
            ],
            "collaboration_feedback": {"developer1": [{"satisfaction": 4.0}]},
        }

        features = self.design_enhanced_features(
            sample_task, sample_developer, sample_env
        )
        return list(features.keys())

    def validate_features(self, features: Dict[str, float]) -> Dict[str, Any]:
        """特徴量の妥当性を検証

        Args:
            features: 特徴量辞書

        Returns:
            検証結果辞書
        """
        validation_results = {
            "valid_features": [],
            "invalid_features": [],
            "warnings": [],
            "statistics": {},
        }

        for name, value in features.items():
            # NaN、無限値のチェック
            if np.isnan(value) or np.isinf(value):
                validation_results["invalid_features"].append(
                    {"name": name, "value": value, "issue": "NaN or Inf value"}
                )
            else:
                validation_results["valid_features"].append(name)

            # 確率・比率特徴量の範囲チェック
            if any(
                keyword in name
                for keyword in [
                    "ratio",
                    "rate",
                    "probability",
                    "compatibility",
                    "similarity",
                ]
            ):
                if value < 0 or value > 1:
                    validation_results["warnings"].append(
                        {
                            "name": name,
                            "value": value,
                            "warning": "Probability/ratio value outside [0,1] range",
                        }
                    )

        # 統計情報
        valid_values = [
            v for k, v in features.items() if k in validation_results["valid_features"]
        ]
        if valid_values:
            validation_results["statistics"] = {
                "count": len(valid_values),
                "mean": float(np.mean(valid_values)),
                "std": float(np.std(valid_values)),
                "min": float(np.min(valid_values)),
                "max": float(np.max(valid_values)),
            }

        return validation_results
