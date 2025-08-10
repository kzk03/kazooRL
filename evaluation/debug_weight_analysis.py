#!/usr/bin/env python3
"""
軽量版重み分析デバッグ
"""

import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class PPOPolicyNetwork(nn.Module):
    """PPOポリシーネットワークの再構築"""

    def __init__(self, input_dim=64, hidden_dim=128):
        super(PPOPolicyNetwork, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Softmax(dim=-1),
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

    def get_action_score(self, x):
        with torch.no_grad():
            action_probs, value = self.forward(x)
            score = torch.max(action_probs).item()
            return score


def is_bot(username: str) -> bool:
    """ユーザー名がBotかどうか判定"""
    bot_indicators = [
        "[bot]",
        "bot",
        "dependabot",
        "renovate",
        "greenkeeper",
        "codecov",
        "travis",
        "circleci",
        "github-actions",
        "automated",
    ]
    username_lower = username.lower()
    return any(indicator in username_lower for indicator in bot_indicators)


class LightweightWeightAnalyzer:
    """軽量版重み分析システム"""

    def __init__(self, model_dir: str, test_data_path: str):
        print("🔧 軽量版システム初期化中...")
        self.model_dir = model_dir
        self.test_data_path = test_data_path
        self.models = {}
        self.author_contributions = {}

        self._load_data()
        self._load_models()
        print(f"   初期化完了: {len(self.models)}モデル, {len(self.tasks)}タスク")

    def _load_data(self):
        """データ読み込み"""
        with open(self.test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        self.tasks = []
        self.ground_truth = []

        for task in test_data:
            author = task.get("author", {})
            if author and isinstance(author, dict):
                author_login = author.get("login", "")
                if author_login and not is_bot(author_login):
                    self.tasks.append(task)
                    self.ground_truth.append(author_login)

        self.author_contributions = Counter(self.ground_truth)

    def _load_models(self):
        """モデル読み込み"""
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith(".pth")]
        all_trained_agents = [
            f.replace("agent_", "").replace(".pth", "") for f in model_files
        ]

        human_agents = [agent for agent in all_trained_agents if not is_bot(agent)]
        actual_set = set(self.ground_truth)
        overlapping_agents = actual_set.intersection(set(human_agents))

        for agent_name in list(overlapping_agents)[:20]:  # 最初の20人だけ
            model_path = os.path.join(self.model_dir, f"agent_{agent_name}.pth")
            try:
                model_data = torch.load(
                    model_path, map_location="cpu", weights_only=False
                )
                policy_network = PPOPolicyNetwork()
                policy_network.load_state_dict(model_data["policy_state_dict"])
                policy_network.eval()
                self.models[agent_name] = policy_network
            except Exception:
                continue

    def _extract_task_features(self, task: Dict) -> torch.Tensor:
        """簡易タスク特徴量抽出"""
        features = []

        title = task.get("title", "") or ""
        body = task.get("body", "") or ""

        # 基本特徴量
        features.extend(
            [
                len(title),
                len(body),
                len(title.split()),
                len(body.split()),
                1 if "bug" in title.lower() else 0,
                1 if "feature" in title.lower() else 0,
                1 if "doc" in title.lower() else 0,
            ]
        )

        # パディング
        while len(features) < 64:
            features.append(0.0)
        features = features[:64]

        return torch.tensor(features, dtype=torch.float32)

    def simple_weight_test(self, sample_size: int = 50):
        """シンプルな重み分析テスト"""
        print(f"\n🔍 シンプル重み分析開始 (サンプル数: {sample_size})")

        available_agents = set(self.models.keys())

        # 評価用タスク選択
        eval_tasks = []
        eval_ground_truth = []

        for task, author in zip(
            self.tasks[: sample_size * 3], self.ground_truth[: sample_size * 3]
        ):
            if author in available_agents and len(eval_tasks) < sample_size:
                eval_tasks.append(task)
                eval_ground_truth.append(author)

        print(f"   実際の評価タスク数: {len(eval_tasks)}")

        # 重みパターン（シンプル版）
        weight_patterns = {
            "balanced": {"ppo": 0.5, "contribution": 0.5},
            "ppo_heavy": {"ppo": 0.8, "contribution": 0.2},
            "contribution_heavy": {"ppo": 0.2, "contribution": 0.8},
            "ppo_only": {"ppo": 1.0, "contribution": 0.0},
            "contribution_only": {"ppo": 0.0, "contribution": 1.0},
        }

        results = {}

        for pattern_name, weights in weight_patterns.items():
            print(f"\n   {pattern_name}パターン評価中...")
            correct_count = 0

            # 修正: zip()を使って正しくペアにする
            for task, actual_author in zip(eval_tasks, eval_ground_truth):
                try:
                    task_features = self._extract_task_features(task)
                    agent_scores = {}

                    for agent_name, model in self.models.items():
                        # PPOスコア
                        ppo_score = model.get_action_score(task_features)

                        # 貢献量スコア
                        contribution = self.author_contributions.get(agent_name, 0)
                        contribution_score = min(contribution / 100.0, 1.0)

                        # 重み付き最終スコア
                        final_score = (
                            weights["ppo"] * ppo_score
                            + weights["contribution"] * contribution_score
                        )

                        agent_scores[agent_name] = final_score

                    # Top-1推薦
                    if agent_scores:
                        top_agent = max(agent_scores.items(), key=lambda x: x[1])[0]
                        if top_agent == actual_author:
                            correct_count += 1

                except Exception as e:
                    print(f"     エラー: {e}")
                    continue

            accuracy = correct_count / len(eval_tasks) if eval_tasks else 0
            results[pattern_name] = {
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_count": len(eval_tasks),
                "weights": weights,
            }

            print(f"     精度: {accuracy:.3f} ({accuracy*100:.1f}%)")
            print(f"     正解数: {correct_count}/{len(eval_tasks)}")

        # 結果表示
        print(f"\n## 📊 結果サマリー")
        print("=" * 40)

        sorted_results = sorted(
            results.items(), key=lambda x: x[1]["accuracy"], reverse=True
        )

        print("| パターン名 | 精度 | PPO重み | 貢献量重み |")
        print("|------------|------|---------|------------|")

        for pattern_name, result in sorted_results:
            accuracy = result["accuracy"]
            ppo_weight = result["weights"]["ppo"]
            contrib_weight = result["weights"]["contribution"]
            print(
                f"| {pattern_name} | {accuracy:.3f} ({accuracy*100:.1f}%) | {ppo_weight:.1f} | {contrib_weight:.1f} |"
            )

        # 最高精度パターン
        best_pattern_name, best_result = sorted_results[0]
        print(f"\n🏆 最高精度: {best_pattern_name}")
        print(f"   精度: {best_result['accuracy']*100:.1f}%")
        print(f"   PPO重み: {best_result['weights']['ppo']}")
        print(f"   貢献量重み: {best_result['weights']['contribution']}")

        return results


def main():
    """メイン実行"""
    print("🚀 軽量版重み分析システム")
    print("=" * 40)

    try:
        # システム初期化
        analyzer = LightweightWeightAnalyzer(
            model_dir="models/improved_rl/final_models",
            test_data_path="data/backlog_test_2023.json",
        )

        # 重み分析実行
        results = analyzer.simple_weight_test(sample_size=30)

        print(f"\n✅ 分析完了！")

    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
