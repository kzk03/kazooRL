"""
開発者特徴量設計器
================

既存開発者特徴量の改良と新規特徴量の追加機能を提供します。
専門性、活動パターン、品質の特徴量を実装します。
"""

import logging
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DeveloperFeatureDesigner:
    """開発者特徴量設計器
    
    既存開発者特徴量の改良（対数変換、比率計算、効率性指標）を実装。
    専門性、活動パターン、品質の新規特徴量を追加。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 設定辞書（言語リスト、ドメインリストなど）
        """
        self.config = config or {}
        
        # プログラミング言語リスト
        self.programming_languages = self.config.get('programming_languages', [
            'python', 'javascript', 'java', 'c++', 'c#', 'go', 'rust', 'typescript',
            'php', 'ruby', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql'
        ])
        
        # 技術ドメイン
        self.technical_domains = self.config.get('technical_domains', [
            'web', 'mobile', 'backend', 'frontend', 'database', 'devops', 'ml',
            'ai', 'security', 'testing', 'ui/ux', 'api', 'cloud', 'embedded'
        ])
        
        # フレームワーク・ライブラリ
        self.frameworks = self.config.get('frameworks', [
            'react', 'vue', 'angular', 'django', 'flask', 'spring', 'express',
            'tensorflow', 'pytorch', 'pandas', 'numpy', 'docker', 'kubernetes'
        ])
        
        # 品質指標の重み
        self.quality_weights = self.config.get('quality_weights', {
            'pr_merge_rate': 0.3,
            'review_approval_rate': 0.25,
            'bug_introduction_rate': 0.25,
            'code_review_quality': 0.2
        })
        
        logger.info(f"DeveloperFeatureDesigner初期化完了")
    
    def design_enhanced_features(self, developer_data: Dict[str, Any], 
                                env_context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """強化された開発者特徴量を設計
        
        Args:
            developer_data: 開発者データ辞書
            env_context: 環境コンテキスト（プロジェクト情報、他の開発者情報など）
            
        Returns:
            強化された特徴量辞書
        """
        features = {}
        
        # 既存特徴量の改良
        features.update(self._improve_existing_features(developer_data))
        
        # 専門性特徴量
        features.update(self._extract_expertise_features(developer_data))
        
        # 活動パターン特徴量
        features.update(self._extract_activity_pattern_features(developer_data))
        
        # 品質特徴量
        features.update(self._extract_quality_features(developer_data))
        
        logger.debug(f"開発者特徴量設計完了: {len(features)}個の特徴量")
        return features
    
    def _improve_existing_features(self, developer_data: Dict[str, Any]) -> Dict[str, float]:
        """既存特徴量の改良
        
        Args:
            developer_data: 開発者データ
            
        Returns:
            改良された既存特徴量
        """
        features = {}
        
        # 基本的な開発者情報
        total_commits = developer_data.get('total_commits', 0)
        total_prs = developer_data.get('total_prs', 0)
        total_issues = developer_data.get('total_issues', 0)
        total_lines_changed = developer_data.get('total_lines_changed', 0)
        collaboration_network_size = developer_data.get('collaboration_network_size', 0)
        comment_interactions = developer_data.get('comment_interactions', 0)
        
        # 1. 活動量の対数変換（外れ値の影響を軽減）
        features['dev_total_commits_log'] = float(np.log1p(total_commits))
        features['dev_total_prs_log'] = float(np.log1p(total_prs))
        features['dev_total_issues_log'] = float(np.log1p(total_issues))
        features['dev_total_lines_changed_log'] = float(np.log1p(total_lines_changed))
        
        # 2. 協力ネットワークの対数変換
        features['dev_collaboration_network_log'] = float(np.log1p(collaboration_network_size))
        features['dev_comment_interactions_log'] = float(np.log1p(comment_interactions))
        
        # 3. 効率性指標（比率計算）
        if total_commits > 0:
            features['dev_lines_per_commit'] = float(total_lines_changed / total_commits)
            features['dev_prs_per_commit_ratio'] = float(total_prs / total_commits)
        else:
            features['dev_lines_per_commit'] = 0.0
            features['dev_prs_per_commit_ratio'] = 0.0
        
        if total_prs > 0:
            features['dev_commits_per_pr'] = float(total_commits / total_prs)
        else:
            features['dev_commits_per_pr'] = 0.0
        
        # 4. 社会性指標
        total_activity = total_commits + total_prs + total_issues
        if total_activity > 0:
            features['dev_collaboration_ratio'] = float(collaboration_network_size / total_activity)
            features['dev_comment_activity_ratio'] = float(comment_interactions / total_activity)
        else:
            features['dev_collaboration_ratio'] = 0.0
            features['dev_comment_activity_ratio'] = 0.0
        
        # 5. 活動多様性
        activity_types = [total_commits, total_prs, total_issues]
        activity_entropy = self._calculate_entropy(activity_types)
        features['dev_activity_diversity'] = float(activity_entropy)
        
        # 6. 最近の活動レベル
        recent_activity = developer_data.get('recent_activity_count', 0)
        features['dev_recent_activity_log'] = float(np.log1p(recent_activity))
        
        # 現在の作業負荷
        current_workload = developer_data.get('current_workload', 0)
        features['dev_current_workload_log'] = float(np.log1p(current_workload))
        
        return features
    
    def _extract_expertise_features(self, developer_data: Dict[str, Any]) -> Dict[str, float]:
        """専門性特徴量の抽出
        
        Args:
            developer_data: 開発者データ
            
        Returns:
            専門性特徴量辞書
        """
        features = {}
        
        # 1. 主要言語強度
        languages = developer_data.get('languages', {})
        if languages:
            # 最も使用している言語の比率
            total_lang_usage = sum(languages.values())
            if total_lang_usage > 0:
                max_lang_usage = max(languages.values())
                features['dev_primary_language_strength'] = float(max_lang_usage / total_lang_usage)
                
                # 上位3言語の集中度
                sorted_langs = sorted(languages.values(), reverse=True)
                top3_usage = sum(sorted_langs[:3])
                features['dev_top3_language_concentration'] = float(top3_usage / total_lang_usage)
            else:
                features['dev_primary_language_strength'] = 0.0
                features['dev_top3_language_concentration'] = 0.0
            
            # 言語多様性（エントロピー）
            lang_entropy = self._calculate_entropy(list(languages.values()))
            features['dev_language_diversity'] = float(lang_entropy)
            
            # 知っている言語数
            features['dev_language_count'] = float(len(languages))
        else:
            features['dev_primary_language_strength'] = 0.0
            features['dev_top3_language_concentration'] = 0.0
            features['dev_language_diversity'] = 0.0
            features['dev_language_count'] = 0.0
        
        # 2. ドメイン専門性
        touched_files = developer_data.get('touched_files', [])
        domain_scores = self._calculate_domain_expertise(touched_files)
        
        for domain, score in domain_scores.items():
            features[f'dev_domain_{domain}_expertise'] = float(score)
        
        # 最高ドメイン専門性
        if domain_scores:
            features['dev_max_domain_expertise'] = float(max(domain_scores.values()))
            features['dev_domain_specialization'] = float(len([s for s in domain_scores.values() if s > 0.1]))
        else:
            features['dev_max_domain_expertise'] = 0.0
            features['dev_domain_specialization'] = 0.0
        
        # 3. 技術多様性
        skills = developer_data.get('skills', [])
        features['dev_skill_count'] = float(len(skills))
        
        # フレームワーク・ライブラリの経験
        framework_experience = 0
        for skill in skills:
            for framework in self.frameworks:
                if framework.lower() in skill.lower():
                    framework_experience += 1
                    break
        
        features['dev_framework_experience_count'] = float(framework_experience)
        
        # 4. 学習速度（新しい技術の習得速度の推定）
        # 最近追加されたスキル数 / 全スキル数
        recent_skills = developer_data.get('recent_skills', [])
        if len(skills) > 0:
            features['dev_learning_rate'] = float(len(recent_skills) / len(skills))
        else:
            features['dev_learning_rate'] = 0.0
        
        # 技術トレンドへの適応度
        modern_tech_keywords = ['ai', 'ml', 'cloud', 'microservices', 'containerization', 'devops']
        modern_tech_count = 0
        for skill in skills:
            for keyword in modern_tech_keywords:
                if keyword in skill.lower():
                    modern_tech_count += 1
                    break
        
        features['dev_modern_tech_adoption'] = float(modern_tech_count)
        
        return features
    
    def _extract_activity_pattern_features(self, developer_data: Dict[str, Any]) -> Dict[str, float]:
        """活動パターン特徴量の抽出
        
        Args:
            developer_data: 開発者データ
            
        Returns:
            活動パターン特徴量辞書
        """
        features = {}
        
        # 1. タイムゾーン推定（活動時間から）
        activity_hours = developer_data.get('activity_hours', [])
        if activity_hours:
            # 最も活発な時間帯
            hour_counts = Counter(activity_hours)
            peak_hour = hour_counts.most_common(1)[0][0] if hour_counts else 12
            features['dev_peak_activity_hour'] = float(peak_hour)
            
            # 活動時間の分散（一貫性の指標）
            features['dev_activity_hour_variance'] = float(np.var(activity_hours))
            
            # 夜間活動比率（22時-6時）
            night_hours = [h for h in activity_hours if h >= 22 or h <= 6]
            features['dev_night_activity_ratio'] = float(len(night_hours) / len(activity_hours))
            
            # 営業時間活動比率（9時-17時）
            business_hours = [h for h in activity_hours if 9 <= h <= 17]
            features['dev_business_hours_ratio'] = float(len(business_hours) / len(activity_hours))
        else:
            features['dev_peak_activity_hour'] = 12.0  # デフォルト
            features['dev_activity_hour_variance'] = 0.0
            features['dev_night_activity_ratio'] = 0.0
            features['dev_business_hours_ratio'] = 0.5
        
        # 2. 平日・週末活動パターン
        activity_days = developer_data.get('activity_days', [])  # 0=月曜, 6=日曜
        if activity_days:
            weekday_activities = [d for d in activity_days if d < 5]  # 月-金
            weekend_activities = [d for d in activity_days if d >= 5]  # 土日
            
            total_activities = len(activity_days)
            features['dev_weekday_activity_ratio'] = float(len(weekday_activities) / total_activities)
            features['dev_weekend_activity_ratio'] = float(len(weekend_activities) / total_activities)
            
            # 活動日の一貫性
            day_counts = Counter(activity_days)
            day_entropy = self._calculate_entropy(list(day_counts.values()))
            features['dev_activity_day_consistency'] = float(1.0 / (1.0 + day_entropy))
        else:
            features['dev_weekday_activity_ratio'] = 0.7  # デフォルト
            features['dev_weekend_activity_ratio'] = 0.3
            features['dev_activity_day_consistency'] = 0.5
        
        # 3. 応答時間パターン
        response_times = developer_data.get('response_times', [])  # 時間単位
        if response_times:
            features['dev_avg_response_time'] = float(np.mean(response_times))
            features['dev_response_time_std'] = float(np.std(response_times))
            features['dev_median_response_time'] = float(np.median(response_times))
            
            # 迅速応答率（1時間以内）
            quick_responses = [t for t in response_times if t <= 1.0]
            features['dev_quick_response_ratio'] = float(len(quick_responses) / len(response_times))
            
            # 応答時間の対数変換
            features['dev_avg_response_time_log'] = float(np.log1p(np.mean(response_times)))
        else:
            features['dev_avg_response_time'] = 24.0  # デフォルト24時間
            features['dev_response_time_std'] = 12.0
            features['dev_median_response_time'] = 12.0
            features['dev_quick_response_ratio'] = 0.2
            features['dev_avg_response_time_log'] = float(np.log1p(24.0))
        
        # 4. 活動一貫性スコア
        # 複数の一貫性指標を組み合わせ
        consistency_factors = [
            features.get('dev_activity_day_consistency', 0.5),
            1.0 / (1.0 + features.get('dev_activity_hour_variance', 1.0) / 100),  # 正規化
            1.0 / (1.0 + features.get('dev_response_time_std', 12.0) / 24)  # 正規化
        ]
        
        features['dev_overall_consistency_score'] = float(np.mean(consistency_factors))
        
        return features
    
    def _extract_quality_features(self, developer_data: Dict[str, Any]) -> Dict[str, float]:
        """品質特徴量の抽出
        
        Args:
            developer_data: 開発者データ
            
        Returns:
            品質特徴量辞書
        """
        features = {}
        
        # 1. PRマージ率
        total_prs = developer_data.get('total_prs', 0)
        merged_prs = developer_data.get('merged_prs', 0)
        
        if total_prs > 0:
            features['dev_pr_merge_rate'] = float(merged_prs / total_prs)
        else:
            features['dev_pr_merge_rate'] = 0.0
        
        # 2. レビュー承認率
        reviews_given = developer_data.get('reviews_given', 0)
        approved_reviews = developer_data.get('approved_reviews', 0)
        
        if reviews_given > 0:
            features['dev_review_approval_rate'] = float(approved_reviews / reviews_given)
        else:
            features['dev_review_approval_rate'] = 0.0
        
        # 3. バグ導入率（逆指標）
        commits_with_bugs = developer_data.get('commits_with_bugs', 0)
        total_commits = developer_data.get('total_commits', 0)
        
        if total_commits > 0:
            bug_rate = commits_with_bugs / total_commits
            features['dev_bug_introduction_rate'] = float(bug_rate)
            features['dev_code_reliability'] = float(1.0 - bug_rate)  # 信頼性指標
        else:
            features['dev_bug_introduction_rate'] = 0.0
            features['dev_code_reliability'] = 1.0
        
        # 4. コードレビュー品質
        review_comments = developer_data.get('review_comments', 0)
        helpful_reviews = developer_data.get('helpful_reviews', 0)
        
        if review_comments > 0:
            features['dev_review_helpfulness_ratio'] = float(helpful_reviews / review_comments)
        else:
            features['dev_review_helpfulness_ratio'] = 0.0
        
        # レビューの詳細度（コメント数/レビュー数）
        if reviews_given > 0:
            features['dev_review_detail_level'] = float(review_comments / reviews_given)
        else:
            features['dev_review_detail_level'] = 0.0
        
        # 5. テスト品質
        test_files_touched = developer_data.get('test_files_touched', 0)
        total_files_touched = developer_data.get('total_files_touched', 0)
        
        if total_files_touched > 0:
            features['dev_test_coverage_ratio'] = float(test_files_touched / total_files_touched)
        else:
            features['dev_test_coverage_ratio'] = 0.0
        
        # 6. ドキュメント品質
        doc_files_touched = developer_data.get('doc_files_touched', 0)
        if total_files_touched > 0:
            features['dev_documentation_ratio'] = float(doc_files_touched / total_files_touched)
        else:
            features['dev_documentation_ratio'] = 0.0
        
        # 7. 総合品質スコア
        quality_components = [
            features['dev_pr_merge_rate'] * self.quality_weights['pr_merge_rate'],
            features['dev_review_approval_rate'] * self.quality_weights['review_approval_rate'],
            (1.0 - features['dev_bug_introduction_rate']) * self.quality_weights['bug_introduction_rate'],
            features['dev_review_helpfulness_ratio'] * self.quality_weights['code_review_quality']
        ]
        
        features['dev_overall_quality_score'] = float(sum(quality_components))
        
        # 8. 経験に基づく品質指標
        years_experience = developer_data.get('years_experience', 0)
        if years_experience > 0:
            features['dev_quality_per_experience'] = float(features['dev_overall_quality_score'] / years_experience)
        else:
            features['dev_quality_per_experience'] = features['dev_overall_quality_score']
        
        return features
    
    def _calculate_entropy(self, values: List[Union[int, float]]) -> float:
        """エントロピーを計算（多様性の指標）
        
        Args:
            values: 値のリスト
            
        Returns:
            エントロピー値
        """
        if not values or sum(values) == 0:
            return 0.0
        
        total = sum(values)
        probabilities = [v / total for v in values if v > 0]
        
        entropy = -sum(p * np.log2(p) for p in probabilities)
        return entropy
    
    def _calculate_domain_expertise(self, touched_files: List[str]) -> Dict[str, float]:
        """ファイルパスからドメイン専門性を計算
        
        Args:
            touched_files: 触れたファイルのパスリスト
            
        Returns:
            ドメイン別専門性スコア
        """
        domain_counts = {domain: 0 for domain in self.technical_domains}
        
        for file_path in touched_files:
            file_path_lower = file_path.lower()
            
            # ファイルパスやファイル名からドメインを推定
            for domain in self.technical_domains:
                if domain in file_path_lower:
                    domain_counts[domain] += 1
                elif domain == 'frontend' and any(term in file_path_lower for term in ['ui', 'view', 'component', 'css', 'html']):
                    domain_counts[domain] += 1
                elif domain == 'backend' and any(term in file_path_lower for term in ['api', 'server', 'service', 'controller']):
                    domain_counts[domain] += 1
                elif domain == 'database' and any(term in file_path_lower for term in ['db', 'sql', 'migration', 'schema']):
                    domain_counts[domain] += 1
                elif domain == 'testing' and any(term in file_path_lower for term in ['test', 'spec', '__test__']):
                    domain_counts[domain] += 1
        
        # 正規化
        total_files = len(touched_files)
        if total_files > 0:
            domain_scores = {domain: count / total_files for domain, count in domain_counts.items()}
        else:
            domain_scores = {domain: 0.0 for domain in self.technical_domains}
        
        return domain_scores
    
    def get_feature_names(self) -> List[str]:
        """設計される特徴量名のリストを取得
        
        Returns:
            特徴量名のリスト
        """
        # サンプルデータで特徴量名を取得
        sample_developer = {
            'total_commits': 100,
            'total_prs': 50,
            'total_issues': 20,
            'total_lines_changed': 5000,
            'collaboration_network_size': 15,
            'comment_interactions': 200,
            'recent_activity_count': 10,
            'current_workload': 3,
            'languages': {'python': 60, 'javascript': 30, 'sql': 10},
            'touched_files': ['src/api/views.py', 'frontend/components/App.js', 'tests/test_api.py'],
            'skills': ['python', 'django', 'react', 'postgresql'],
            'recent_skills': ['docker'],
            'activity_hours': [9, 10, 14, 15, 16],
            'activity_days': [0, 1, 2, 3, 4],
            'response_times': [2.5, 1.0, 4.0, 0.5],
            'merged_prs': 45,
            'reviews_given': 30,
            'approved_reviews': 25,
            'commits_with_bugs': 5,
            'review_comments': 100,
            'helpful_reviews': 80,
            'test_files_touched': 20,
            'total_files_touched': 100,
            'doc_files_touched': 10,
            'years_experience': 3
        }
        
        features = self.design_enhanced_features(sample_developer)
        return list(features.keys())
    
    def validate_features(self, features: Dict[str, float]) -> Dict[str, Any]:
        """特徴量の妥当性を検証
        
        Args:
            features: 特徴量辞書
            
        Returns:
            検証結果辞書
        """
        validation_results = {
            'valid_features': [],
            'invalid_features': [],
            'warnings': [],
            'statistics': {}
        }
        
        for name, value in features.items():
            # NaN、無限値のチェック
            if np.isnan(value) or np.isinf(value):
                validation_results['invalid_features'].append({
                    'name': name,
                    'value': value,
                    'issue': 'NaN or Inf value'
                })
            else:
                validation_results['valid_features'].append(name)
            
            # 比率特徴量の範囲チェック（0-1）
            if 'ratio' in name or 'rate' in name:
                if value < 0 or value > 1:
                    validation_results['warnings'].append({
                        'name': name,
                        'value': value,
                        'warning': 'Ratio/rate value outside [0,1] range'
                    })
        
        # 統計情報
        valid_values = [v for k, v in features.items() if k in validation_results['valid_features']]
        if valid_values:
            validation_results['statistics'] = {
                'count': len(valid_values),
                'mean': float(np.mean(valid_values)),
                'std': float(np.std(valid_values)),
                'min': float(np.min(valid_values)),
                'max': float(np.max(valid_values))
            }
        
        return validation_results