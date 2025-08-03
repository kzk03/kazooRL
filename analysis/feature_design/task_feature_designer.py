"""
タスク特徴量設計器
================

既存タスク特徴量の改良と新規特徴量の追加機能を提供します。
緊急度、複雑度、社会的注目度の特徴量を実装します。
"""

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class TaskFeatureDesigner:
    """タスク特徴量設計器
    
    既存タスク特徴量の改良（対数変換、正規化、複雑度計算）を実装。
    緊急度、複雑度、社会的注目度の新規特徴量を追加。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 設定辞書（優先度ラベル、技術用語リストなど）
        """
        self.config = config or {}
        
        # デフォルト設定
        self.priority_labels = self.config.get('priority_labels', {
            'critical': 5, 'high': 4, 'medium': 3, 'low': 2, 'trivial': 1
        })
        
        self.technical_terms = self.config.get('technical_terms', [
            'API', 'database', 'algorithm', 'performance', 'security', 'authentication',
            'authorization', 'encryption', 'optimization', 'refactor', 'architecture',
            'framework', 'library', 'dependency', 'configuration', 'deployment',
            'testing', 'debugging', 'logging', 'monitoring', 'scalability'
        ])
        
        self.urgency_keywords = self.config.get('urgency_keywords', [
            'urgent', 'critical', 'asap', 'immediately', 'emergency', 'hotfix',
            'blocker', 'blocking', 'deadline', 'milestone', 'release'
        ])
        
        logger.info(f"TaskFeatureDesigner初期化完了")
    
    def design_enhanced_features(self, task_data: Dict[str, Any], 
                                env_context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """強化されたタスク特徴量を設計
        
        Args:
            task_data: タスクデータ辞書
            env_context: 環境コンテキスト（他のタスク、開発者情報など）
            
        Returns:
            強化された特徴量辞書
        """
        features = {}
        
        # 既存特徴量の改良
        features.update(self._improve_existing_features(task_data))
        
        # 緊急度特徴量
        features.update(self._extract_urgency_features(task_data, env_context))
        
        # 複雑度特徴量
        features.update(self._extract_complexity_features(task_data))
        
        # 社会的注目度特徴量
        features.update(self._extract_social_attention_features(task_data))
        
        logger.debug(f"タスク特徴量設計完了: {len(features)}個の特徴量")
        return features
    
    def _improve_existing_features(self, task_data: Dict[str, Any]) -> Dict[str, float]:
        """既存特徴量の改良
        
        Args:
            task_data: タスクデータ
            
        Returns:
            改良された既存特徴量
        """
        features = {}
        
        # 基本的なタスク情報
        title = task_data.get('title', '')
        body = task_data.get('body', '')
        comments = task_data.get('comments', 0)
        labels = task_data.get('labels', [])
        created_at = task_data.get('created_at')
        updated_at = task_data.get('updated_at')
        
        # 1. テキスト長の対数変換（外れ値の影響を軽減）
        text_length = len(title) + len(body)
        features['task_text_length_log'] = float(np.log1p(text_length))
        
        # 2. コメント数の対数変換
        features['task_comments_log'] = float(np.log1p(comments))
        
        # 3. 活動度の改良（最終更新からの経過時間の対数変換）
        if updated_at:
            try:
                if isinstance(updated_at, str):
                    updated_time = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                else:
                    updated_time = updated_at
                
                days_since_update = (datetime.now() - updated_time.replace(tzinfo=None)).days
                features['task_days_since_update_log'] = float(np.log1p(max(0, days_since_update)))
            except Exception as e:
                logger.warning(f"日付解析エラー: {e}")
                features['task_days_since_update_log'] = 0.0
        else:
            features['task_days_since_update_log'] = 0.0
        
        # 4. タスク年齢（作成からの経過時間）
        if created_at:
            try:
                if isinstance(created_at, str):
                    created_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                else:
                    created_time = created_at
                
                task_age_days = (datetime.now() - created_time.replace(tzinfo=None)).days
                features['task_age_days_log'] = float(np.log1p(max(0, task_age_days)))
            except Exception as e:
                logger.warning(f"作成日付解析エラー: {e}")
                features['task_age_days_log'] = 0.0
        else:
            features['task_age_days_log'] = 0.0
        
        # 5. ラベル数
        features['task_label_count'] = float(len(labels))
        
        # 6. タイトル・本文の比率
        title_length = len(title)
        if text_length > 0:
            features['task_title_body_ratio'] = float(title_length / text_length)
        else:
            features['task_title_body_ratio'] = 0.0
        
        # 7. コードブロック密度（改良版）
        code_blocks = body.count('```')
        if text_length > 0:
            features['task_code_density'] = float(code_blocks / text_length * 1000)  # 1000文字あたり
        else:
            features['task_code_density'] = 0.0
        
        return features
    
    def _extract_urgency_features(self, task_data: Dict[str, Any], 
                                 env_context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """緊急度特徴量の抽出
        
        Args:
            task_data: タスクデータ
            env_context: 環境コンテキスト
            
        Returns:
            緊急度特徴量辞書
        """
        features = {}
        
        title = task_data.get('title', '').lower()
        body = task_data.get('body', '').lower()
        labels = [label.lower() for label in task_data.get('labels', [])]
        
        # 1. 優先度ラベルベースの緊急度
        priority_score = 0.0
        for label in labels:
            for priority_label, score in self.priority_labels.items():
                if priority_label in label:
                    priority_score = max(priority_score, score)
        features['task_priority_label_score'] = float(priority_score)
        
        # 2. 緊急度キーワードの出現頻度
        urgency_count = 0
        text_content = title + ' ' + body
        for keyword in self.urgency_keywords:
            urgency_count += text_content.count(keyword.lower())
        features['task_urgency_keyword_count'] = float(urgency_count)
        
        # 3. 緊急度キーワード密度
        text_length = len(text_content)
        if text_length > 0:
            features['task_urgency_keyword_density'] = float(urgency_count / text_length * 1000)
        else:
            features['task_urgency_keyword_density'] = 0.0
        
        # 4. 期限関連の特徴量
        deadline_keywords = ['deadline', 'due', 'by', 'until', 'before']
        deadline_mentions = sum(text_content.count(keyword) for keyword in deadline_keywords)
        features['task_deadline_mentions'] = float(deadline_mentions)
        
        # 5. マイルストーン関連
        milestone_keywords = ['milestone', 'release', 'version', 'sprint']
        milestone_mentions = sum(text_content.count(keyword) for keyword in milestone_keywords)
        features['task_milestone_mentions'] = float(milestone_mentions)
        
        # 6. ブロッキング課題の特徴量
        blocking_keywords = ['block', 'blocker', 'blocking', 'depends', 'dependency']
        blocking_mentions = sum(text_content.count(keyword) for keyword in blocking_keywords)
        features['task_blocking_mentions'] = float(blocking_mentions)
        
        # 7. 環境コンテキストからのブロッキング課題数（利用可能な場合）
        if env_context and 'blocking_issues' in env_context:
            features['task_blocking_issues_count'] = float(len(env_context['blocking_issues']))
        else:
            features['task_blocking_issues_count'] = 0.0
        
        return features
    
    def _extract_complexity_features(self, task_data: Dict[str, Any]) -> Dict[str, float]:
        """複雑度特徴量の抽出
        
        Args:
            task_data: タスクデータ
            
        Returns:
            複雑度特徴量辞書
        """
        features = {}
        
        title = task_data.get('title', '')
        body = task_data.get('body', '')
        text_content = title + ' ' + body
        
        # 1. 技術用語密度
        tech_term_count = 0
        for term in self.technical_terms:
            tech_term_count += text_content.lower().count(term.lower())
        
        text_length = len(text_content)
        if text_length > 0:
            features['task_technical_term_density'] = float(tech_term_count / text_length * 1000)
        else:
            features['task_technical_term_density'] = 0.0
        
        # 2. 参照リンク数
        # URL、GitHub issue参照、PR参照などを検出
        url_pattern = r'https?://[^\s]+'
        github_ref_pattern = r'#\d+'
        pr_pattern = r'PR\s*#?\d+'
        
        url_count = len(re.findall(url_pattern, text_content))
        github_ref_count = len(re.findall(github_ref_pattern, text_content))
        pr_count = len(re.findall(pr_pattern, text_content, re.IGNORECASE))
        
        total_references = url_count + github_ref_count + pr_count
        features['task_reference_link_count'] = float(total_references)
        
        # 3. 推定工数（テキスト複雑度ベース）
        # 文の数、単語数、技術用語数を組み合わせて推定
        sentence_count = len(re.split(r'[.!?]+', text_content))
        word_count = len(text_content.split())
        
        # 複雑度スコア計算
        complexity_score = (
            word_count * 0.1 +
            sentence_count * 0.5 +
            tech_term_count * 2.0 +
            total_references * 1.5
        )
        features['task_estimated_effort_score'] = float(complexity_score)
        
        # 4. 依存関係数の推定
        dependency_keywords = [
            'require', 'need', 'depend', 'prerequisite', 'after', 'before',
            'related', 'connect', 'integrate', 'merge', 'combine'
        ]
        dependency_count = sum(text_content.lower().count(keyword) for keyword in dependency_keywords)
        features['task_dependency_mentions'] = float(dependency_count)
        
        # 5. コード複雑度指標
        # コードブロック内の行数、関数定義、クラス定義などを検出
        code_blocks = re.findall(r'```[\s\S]*?```', body)
        total_code_lines = 0
        function_definitions = 0
        class_definitions = 0
        
        for block in code_blocks:
            lines = block.split('\n')
            total_code_lines += len(lines)
            function_definitions += len(re.findall(r'def\s+\w+', block))
            class_definitions += len(re.findall(r'class\s+\w+', block))
        
        features['task_code_lines_count'] = float(total_code_lines)
        features['task_function_definitions'] = float(function_definitions)
        features['task_class_definitions'] = float(class_definitions)
        
        # 6. 質問・疑問の数（複雑度の指標）
        question_count = text_content.count('?')
        question_keywords = ['how', 'what', 'why', 'when', 'where', 'which']
        question_keyword_count = sum(text_content.lower().count(keyword) for keyword in question_keywords)
        
        features['task_question_count'] = float(question_count)
        features['task_question_keyword_count'] = float(question_keyword_count)
        
        return features
    
    def _extract_social_attention_features(self, task_data: Dict[str, Any]) -> Dict[str, float]:
        """社会的注目度特徴量の抽出
        
        Args:
            task_data: タスクデータ
            
        Returns:
            社会的注目度特徴量辞書
        """
        features = {}
        
        # 1. ウォッチャー数
        watchers = task_data.get('watchers', 0)
        features['task_watchers_count'] = float(watchers)
        features['task_watchers_log'] = float(np.log1p(watchers))
        
        # 2. リアクション数
        reactions = task_data.get('reactions', {})
        if isinstance(reactions, dict):
            total_reactions = sum(reactions.values())
        else:
            total_reactions = 0
        
        features['task_reactions_total'] = float(total_reactions)
        features['task_reactions_log'] = float(np.log1p(total_reactions))
        
        # 3. 個別リアクション
        reaction_types = ['+1', '-1', 'laugh', 'hooray', 'confused', 'heart', 'rocket', 'eyes']
        for reaction_type in reaction_types:
            count = reactions.get(reaction_type, 0) if isinstance(reactions, dict) else 0
            features[f'task_reaction_{reaction_type.replace("+", "plus").replace("-", "minus")}'] = float(count)
        
        # 4. メンション数
        body = task_data.get('body', '')
        mention_pattern = r'@\w+'
        mention_count = len(re.findall(mention_pattern, body))
        features['task_mention_count'] = float(mention_count)
        
        # 5. 外部参照数（他のissue、PRへの参照）
        external_ref_patterns = [
            r'#\d+',  # Issue/PR参照
            r'https://github\.com/[^/]+/[^/]+/(?:issues|pull)/\d+',  # 外部GitHub参照
        ]
        
        external_ref_count = 0
        for pattern in external_ref_patterns:
            external_ref_count += len(re.findall(pattern, body))
        
        features['task_external_references'] = float(external_ref_count)
        
        # 6. 参加者数（assignees + コメント者の推定）
        assignees = task_data.get('assignees', [])
        assignee_count = len(assignees) if assignees else 0
        
        comments_count = task_data.get('comments', 0)
        # コメント数から参加者数を推定（平均的に1人あたり2-3コメント）
        estimated_participants = max(1, comments_count // 2)
        total_participants = assignee_count + estimated_participants
        
        features['task_assignee_count'] = float(assignee_count)
        features['task_estimated_participants'] = float(total_participants)
        
        # 7. 社会的注目度総合スコア
        attention_score = (
            watchers * 1.0 +
            total_reactions * 2.0 +
            mention_count * 1.5 +
            external_ref_count * 1.0 +
            total_participants * 3.0
        )
        features['task_social_attention_score'] = float(attention_score)
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """設計される特徴量名のリストを取得
        
        Returns:
            特徴量名のリスト
        """
        # サンプルデータで特徴量名を取得
        sample_task = {
            'title': 'Sample Task',
            'body': 'Sample body with ```code``` and @mention',
            'comments': 5,
            'labels': ['bug', 'high'],
            'created_at': '2024-01-01T00:00:00Z',
            'updated_at': '2024-01-02T00:00:00Z',
            'watchers': 10,
            'reactions': {'+1': 3, 'heart': 1},
            'assignees': ['user1']
        }
        
        features = self.design_enhanced_features(sample_task)
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
            
            # 負の値の警告（対数変換特徴量以外）
            if value < 0 and 'log' not in name:
                validation_results['warnings'].append({
                    'name': name,
                    'value': value,
                    'warning': 'Negative value in non-log feature'
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