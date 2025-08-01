#!/usr/bin/env python3
"""
行動空間縮小システム
巨大な行動空間を管理可能なサイズに縮小
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np


class ActionSpaceReducer:
    """行動空間を縮小するためのシステム"""
    
    def __init__(self, config):
        self.config = config
        self.max_candidates = config.get('max_action_candidates', 20)  # 最大候補数
        self.min_candidates = config.get('min_action_candidates', 5)   # 最小候補数
    
    def get_candidate_developers(self, task: Dict, env_state: Dict, 
                               agent_id: str = None) -> List[str]:
        """
        タスクに対する候補開発者を選出
        
        Args:
            task: 対象タスク
            env_state: 環境状態
            agent_id: 現在のエージェントID（除外用）
        
        Returns:
            候補開発者IDのリスト
        """
        all_developers = list(env_state['developers'].keys())
        if agent_id and agent_id in all_developers:
            all_developers.remove(agent_id)
        
        # taskがNoneの場合は全開発者を返す
        if task is None:
            return all_developers[:self.max_candidates]
        
        # 複数の戦略で候補を選出
        candidates = set()
        
        # 1. スキルベース候補
        skill_candidates = self._get_skill_based_candidates(task, env_state)
        candidates.update(skill_candidates[:self.max_candidates//3])
        
        # 2. 活動ベース候補
        activity_candidates = self._get_activity_based_candidates(task, env_state)
        candidates.update(activity_candidates[:self.max_candidates//3])
        
        # 3. 負荷バランス候補
        workload_candidates = self._get_workload_balanced_candidates(env_state)
        candidates.update(workload_candidates[:self.max_candidates//3])
        
        # 4. ランダム候補（探索のため）
        remaining_devs = [dev for dev in all_developers if dev not in candidates]
        if remaining_devs:
            random_count = min(5, len(remaining_devs))
            random_candidates = np.random.choice(remaining_devs, random_count, replace=False)
            candidates.update(random_candidates)
        
        # 最小候補数を保証
        if len(candidates) < self.min_candidates:
            remaining_devs = [dev for dev in all_developers if dev not in candidates]
            additional_needed = self.min_candidates - len(candidates)
            if remaining_devs:
                additional_count = min(additional_needed, len(remaining_devs))
                additional_candidates = np.random.choice(
                    remaining_devs, additional_count, replace=False
                )
                candidates.update(additional_candidates)
        
        # 最大候補数に制限
        candidates_list = list(candidates)[:self.max_candidates]
        
        return candidates_list
    
    def _get_skill_based_candidates(self, task: Dict, env_state: Dict) -> List[str]:
        """スキル適合度に基づく候補選出"""
        developers = env_state['developers']
        skill_scores = []
        
        # タスクの必要スキルを推定
        required_skills = self._extract_required_skills(task)
        
        for dev_id, dev_info in developers.items():
            profile = dev_info['profile']
            
            # 開発者のスキルスコア計算
            skill_score = 0.0
            
            # プログラミング言語スキル
            if 'python' in required_skills:
                python_skill = min(profile.get('python_commits', 0) / 100.0, 1.0)
                skill_score += required_skills['python'] * python_skill
            
            if 'javascript' in required_skills:
                js_skill = min(profile.get('javascript_commits', 0) / 100.0, 1.0)
                skill_score += required_skills['javascript'] * js_skill
            
            # 専門スキル
            if 'debugging' in required_skills:
                debug_skill = min(profile.get('bug_fixes', 0) / 50.0, 1.0)
                skill_score += required_skills['debugging'] * debug_skill
            
            if 'documentation' in required_skills:
                doc_skill = min(profile.get('doc_commits', 0) / 20.0, 1.0)
                skill_score += required_skills['documentation'] * doc_skill
            
            # 総合ランクも考慮
            rank_score = 1.0 - (profile.get('rank', 5000) / 5000.0)
            skill_score += 0.2 * rank_score
            
            skill_scores.append((dev_id, skill_score))
        
        # スキルスコア順にソート
        skill_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [dev_id for dev_id, score in skill_scores]
    
    def _get_activity_based_candidates(self, task: Dict, env_state: Dict) -> List[str]:
        """最近の活動に基づく候補選出"""
        developers = env_state['developers']
        current_time = env_state.get('current_time', datetime.now())
        activity_window = timedelta(days=30)
        
        # taskがNoneの場合は活動度のみで判定
        if task is None:
            activity_scores = []
            for dev_id, dev_info in developers.items():
                profile = dev_info['profile']
                activity_score = profile.get('recent_commits', 0) + profile.get('recent_prs', 0)
                activity_scores.append((dev_id, activity_score))
            
            activity_scores.sort(key=lambda x: x[1], reverse=True)
            return [dev_id for dev_id, score in activity_scores]
        
        activity_scores = []
        
        for dev_id, dev_info in developers.items():
            profile = dev_info['profile']
            
            # 最近の活動スコア
            activity_score = 0.0
            
            # 最近のコミット数
            recent_commits = profile.get('recent_commits_30d', 0)
            activity_score += min(recent_commits / 20.0, 1.0) * 0.4
            
            # 最近のPR数
            recent_prs = profile.get('recent_prs_30d', 0)
            activity_score += min(recent_prs / 5.0, 1.0) * 0.3
            
            # 最近のIssue対応数
            recent_issues = profile.get('recent_issues_30d', 0)
            activity_score += min(recent_issues / 10.0, 1.0) * 0.3
            
            activity_scores.append((dev_id, activity_score))
        
        # 活動スコア順にソート
        activity_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [dev_id for dev_id, score in activity_scores]
    
    def _get_workload_balanced_candidates(self, env_state: Dict) -> List[str]:
        """作業負荷バランスに基づく候補選出"""
        assignments = env_state.get('assignments', {})
        developers = env_state['developers']
        
        workload_scores = []
        
        # 全体の平均作業負荷を計算
        all_workloads = [len(tasks) for tasks in assignments.values()]
        avg_workload = np.mean(all_workloads) if all_workloads else 0
        
        for dev_id in developers.keys():
            current_workload = len(assignments.get(dev_id, set()))
            
            # 作業負荷が少ないほど高いスコア
            workload_score = max(0, avg_workload + 2 - current_workload)
            workload_scores.append((dev_id, workload_score))
        
        # 作業負荷スコア順にソート（負荷が少ない順）
        workload_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [dev_id for dev_id, score in workload_scores]
    
    def _extract_required_skills(self, task: Dict) -> Dict[str, float]:
        """タスクから必要スキルを抽出"""
        required_skills = {}
        
        # taskがNoneの場合のデフォルト処理
        if task is None:
            return {'general': 0.5}  # デフォルトスキル
        
        # ラベルベースのスキル推定
        labels = task.get('labels', [])
        if isinstance(labels, list):
            label_names = [label.get('name', '').lower() if isinstance(label, dict) 
                          else str(label).lower() for label in labels]
            
            for label in label_names:
                if 'bug' in label:
                    required_skills['debugging'] = 0.8
                elif 'enhancement' in label or 'feature' in label:
                    required_skills['python'] = 0.6
                    required_skills['javascript'] = 0.4
                elif 'documentation' in label or 'doc' in label:
                    required_skills['documentation'] = 0.9
        
        # タイトル・本文からのスキル推定
        title = task.get('title', '').lower()
        body = task.get('body', '').lower()
        text = f"{title} {body}"
        
        if 'python' in text:
            required_skills['python'] = required_skills.get('python', 0) + 0.3
        if 'javascript' in text or 'js' in text:
            required_skills['javascript'] = required_skills.get('javascript', 0) + 0.3
        
        # 正規化
        for skill in required_skills:
            required_skills[skill] = min(required_skills[skill], 1.0)
        
        return required_skills


class HierarchicalActionSpace:
    """階層的行動空間"""
    
    def __init__(self, config):
        self.config = config
        self.action_reducer = ActionSpaceReducer(config)
    
    def get_action_space_for_task(self, task: Dict, env_state: Dict) -> Tuple[List[str], Dict]:
        """
        特定のタスクに対する行動空間を取得
        
        Returns:
            (候補開発者リスト, メタデータ)
        """
        candidates = self.action_reducer.get_candidate_developers(task, env_state)
        
        # NO_OP行動を追加
        action_space = candidates + ['NO_OP']
        
        metadata = {
            'total_developers': len(env_state['developers']),
            'candidate_count': len(candidates),
            'reduction_ratio': len(candidates) / len(env_state['developers']),
            'candidates': candidates
        }
        
        return action_space, metadata
    
    def map_action_to_developer(self, action_idx: int, action_space: List[str]) -> str:
        """行動インデックスを開発者IDにマッピング"""
        if 0 <= action_idx < len(action_space):
            return action_space[action_idx]
        else:
            return 'NO_OP'