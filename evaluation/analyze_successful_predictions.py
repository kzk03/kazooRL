#!/usr/bin/env python3
"""
成功した推薦の開発者分析
貢献量との関係を詳しく調査
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


def load_test_data_with_bot_filtering(
    test_data_path: str,
) -> Tuple[List[Dict], List[str]]:
    """テストデータを読み込み、Botを除去"""
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    filtered_tasks = []
    ground_truth_authors = []

    for task in test_data:
        author = task.get("author", {})
        if author and isinstance(author, dict):
            author_login = author.get("login", "")
            if author_login and not is_bot(author_login):
                filtered_tasks.append(task)
                ground_truth_authors.append(author_login)

    return filtered_tasks, ground_truth_authors


def load_sample_models(
    model_dir: str, actual_authors: List[str], max_models: int = 50
) -> Dict[str, PPOPolicyNetwork]:
    """サンプルモデルを読み込み"""
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    all_trained_agents = [
        f.replace("agent_", "").replace(".pth", "") for f in model_files
    ]

    human_trained_agents = [agent for agent in all_trained_agents if not is_bot(agent)]
    actual_set = set(actual_authors)
    human_set = set(human_trained_agents)
    overlapping_agents = actual_set.intersection(human_set)

    loaded_models = {}

    for i, agent_name in enumerate(overlapping_agents):
        if i >= max_models:
            break

        model_path = os.path.join(model_dir, f"agent_{agent_name}.pth")

        try:
            model_data = torch.load(model_path, map_location="cpu", weights_only=False)
            policy_network = PPOPolicyNetwork()
            policy_network.load_state_dict(model_data["policy_state_dict"])
            policy_network.eval()
            loaded_models[agent_name] = policy_network
        except Exception as e:
            continue

    return loaded_models


def extract_task_features_for_model(task: Dict) -> torch.Tensor:
    """モデル用のタスク特徴量を抽出（64次元）"""
    features = []

    title = task.get("title", "") or ""
    body = task.get("body", "") or ""
    labels = task.get("labels", [])

    # 基本特徴量（10次元）
    basic_features = [
        len(title),
        len(body),
        len(title.split()),
        len(body.split()),
        len(labels),
        title.count("?"),
        title.count("!"),
        body.count("\n"),
        len(set(title.lower().split())),
        1 if any(kw in title.lower() for kw in ["bug", "fix", "error"]) else 0,
    ]
    features.extend(basic_features)

    # 日付特徴量（3次元）
    created_at = task.get("created_at", "")
    if created_at:
        try:
            date_parts = created_at.split("T")[0].split("-")
            year, month, day = (
                int(date_parts[0]),
                int(date_parts[1]),
                int(date_parts[2]),
            )
            features.extend([year - 2020, month, day])
        except:
            features.extend([0, 0, 0])
    else:
        features.extend([0, 0, 0])

    # ラベル特徴量（10次元）
    label_text = " ".join(
        [
            str(label) if not isinstance(label, dict) else label.get("name", "")
            for label in labels
        ]
    ).lower()

    important_keywords = [
        "bug",
        "feature",
        "enhancement",
        "documentation",
        "help",
        "question",
        "performance",
        "security",
        "ui",
        "api",
    ]
    for keyword in important_keywords:
        features.append(1 if keyword in label_text else 0)

    # 残りをパディング
    while len(features) < 64:
        features.append(0.0)
    features = features[:64]

    # 正規化
    features = np.array(features, dtype=np.float32)
    features = (features - np.mean(features)) / (np.std(features) + 1e-8)

    return torch.tensor(features, dtype=torch.float32)


def analyze_successful_predictions():
    """成功した推薦の詳細分析"""
    print("🎯 成功した推薦の開発者分析")
    print("=" * 60)

    # データ読み込み
    tasks, ground_truth = load_test_data_with_bot_filtering(
        "data/backlog_test_2023.json"
    )
    trained_models = load_sample_models(
        "models/improved_rl/final_models", ground_truth, 50
    )

    print(f"📊 分析対象: {len(tasks):,}タスク, {len(trained_models)}モデル")

    # 全体の貢献量分析
    author_contribution = Counter(ground_truth)
    total_tasks = len(ground_truth)

    print(f"\n## 1. 全体の開発者貢献量分析")
    print("-" * 40)
    print(f"   総タスク数: {total_tasks:,}")
    print(f"   ユニーク開発者数: {len(author_contribution)}")

    # 貢献量別カテゴリ分け
    high_contributors = []  # 50+ タスク
    medium_contributors = []  # 10-49 タスク
    low_contributors = []  # 1-9 タスク

    for author, count in author_contribution.items():
        if count >= 50:
            high_contributors.append((author, count))
        elif count >= 10:
            medium_contributors.append((author, count))
        else:
            low_contributors.append((author, count))

    print(f"\n   貢献量別分類:")
    print(f"     高貢献者 (50+タスク): {len(high_contributors)}人")
    print(f"     中貢献者 (10-49タスク): {len(medium_contributors)}人")
    print(f"     低貢献者 (1-9タスク): {len(low_contributors)}人")

    # 各カテゴリの詳細
    print(f"\n   高貢献者リスト:")
    for author, count in sorted(high_contributors, key=lambda x: x[1], reverse=True):
        percentage = count / total_tasks * 100
        print(f"     {author}: {count:3d}タスク ({percentage:4.1f}%)")

    print(f"\n   中貢献者リスト:")
    for author, count in sorted(medium_contributors, key=lambda x: x[1], reverse=True)[
        :10
    ]:
        percentage = count / total_tasks * 100
        print(f"     {author}: {count:2d}タスク ({percentage:3.1f}%)")

    # 推薦成功分析
    print(f"\n## 2. 推薦成功分析")
    print("-" * 40)

    available_agents = set(trained_models.keys())
    sample_size = min(1000, len(tasks))  # サンプルサイズ

    successful_predictions = {"top1": [], "top3": [], "top5": []}

    print(f"   分析サンプル数: {sample_size}")

    for i, (task, actual_author) in enumerate(
        tqdm(zip(tasks[:sample_size], ground_truth[:sample_size]), desc="推薦分析中")
    ):
        if actual_author not in available_agents:
            continue

        try:
            # タスク特徴量抽出
            task_features = extract_task_features_for_model(task)

            # 各エージェントでの適合度を計算
            agent_scores = {}
            for agent_name, model in trained_models.items():
                try:
                    score = model.get_action_score(task_features)
                    agent_scores[agent_name] = score
                except:
                    agent_scores[agent_name] = 0.0

            # スコア順にソート
            sorted_agents = sorted(
                agent_scores.items(), key=lambda x: x[1], reverse=True
            )

            # Top-K成功判定
            top1_agents = [agent for agent, _ in sorted_agents[:1]]
            top3_agents = [agent for agent, _ in sorted_agents[:3]]
            top5_agents = [agent for agent, _ in sorted_agents[:5]]

            if actual_author in top1_agents:
                successful_predictions["top1"].append(
                    {
                        "author": actual_author,
                        "contribution": author_contribution[actual_author],
                        "rank": 1,
                        "score": agent_scores[actual_author],
                        "task_id": i,
                    }
                )

            if actual_author in top3_agents:
                actual_rank = next(
                    rank
                    for rank, (agent, _) in enumerate(sorted_agents, 1)
                    if agent == actual_author
                )
                successful_predictions["top3"].append(
                    {
                        "author": actual_author,
                        "contribution": author_contribution[actual_author],
                        "rank": actual_rank,
                        "score": agent_scores[actual_author],
                        "task_id": i,
                    }
                )

            if actual_author in top5_agents:
                actual_rank = next(
                    rank
                    for rank, (agent, _) in enumerate(sorted_agents, 1)
                    if agent == actual_author
                )
                successful_predictions["top5"].append(
                    {
                        "author": actual_author,
                        "contribution": author_contribution[actual_author],
                        "rank": actual_rank,
                        "score": agent_scores[actual_author],
                        "task_id": i,
                    }
                )

        except Exception as e:
            continue

    # 成功した推薦の分析
    print(f"\n## 3. 成功した推薦の詳細分析")
    print("-" * 40)

    for k in ["top1", "top3", "top5"]:
        successes = successful_predictions[k]
        if not successes:
            continue

        print(f"\n### {k.upper()}成功分析 ({len(successes)}件)")

        # 成功した開発者のリスト
        success_authors = [s["author"] for s in successes]
        success_counter = Counter(success_authors)

        print(f"   成功した開発者リスト:")
        for author, count in success_counter.most_common():
            total_contribution = author_contribution[author]
            success_rate = (
                count / total_contribution * 100 if total_contribution > 0 else 0
            )
            print(
                f"     {author}: {count}回成功 / {total_contribution}タスク ({success_rate:.1f}%)"
            )

        # 貢献量別成功率分析
        contribution_analysis = {
            "high": {"successes": 0, "total": 0},
            "medium": {"successes": 0, "total": 0},
            "low": {"successes": 0, "total": 0},
        }

        for success in successes:
            author = success["author"]
            contribution = success["contribution"]

            if contribution >= 50:
                contribution_analysis["high"]["successes"] += 1
            elif contribution >= 10:
                contribution_analysis["medium"]["successes"] += 1
            else:
                contribution_analysis["low"]["successes"] += 1

        # 各カテゴリの総タスク数を計算
        for author, count in author_contribution.items():
            if author in available_agents:
                if count >= 50:
                    contribution_analysis["high"]["total"] += count
                elif count >= 10:
                    contribution_analysis["medium"]["total"] += count
                else:
                    contribution_analysis["low"]["total"] += count

        print(f"\n   貢献量別成功率:")
        for category, data in contribution_analysis.items():
            if data["total"] > 0:
                success_rate = data["successes"] / data["total"] * 100
                category_name = {
                    "high": "高貢献者",
                    "medium": "中貢献者",
                    "low": "低貢献者",
                }[category]
                print(
                    f"     {category_name}: {data['successes']}/{data['total']} ({success_rate:.2f}%)"
                )

    # 貢献量と成功の相関分析
    print(f"\n## 4. 貢献量と推薦成功の相関分析")
    print("-" * 40)

    # Top-3成功での詳細分析
    top3_successes = successful_predictions["top3"]
    if top3_successes:
        contributions = [s["contribution"] for s in top3_successes]
        ranks = [s["rank"] for s in top3_successes]

        print(f"   Top-3成功の統計:")
        print(f"     平均貢献量: {np.mean(contributions):.1f}タスク")
        print(f"     中央値貢献量: {np.median(contributions):.1f}タスク")
        print(f"     最大貢献量: {np.max(contributions)}タスク")
        print(f"     最小貢献量: {np.min(contributions)}タスク")
        print(f"     平均ランク: {np.mean(ranks):.1f}位")

        # 貢献量の分布
        high_contrib_successes = sum(1 for c in contributions if c >= 50)
        medium_contrib_successes = sum(1 for c in contributions if 10 <= c < 50)
        low_contrib_successes = sum(1 for c in contributions if c < 10)

        total_successes = len(contributions)
        print(f"\n   成功の貢献量分布:")
        print(
            f"     高貢献者の成功: {high_contrib_successes}/{total_successes} ({high_contrib_successes/total_successes*100:.1f}%)"
        )
        print(
            f"     中貢献者の成功: {medium_contrib_successes}/{total_successes} ({medium_contrib_successes/total_successes*100:.1f}%)"
        )
        print(
            f"     低貢献者の成功: {low_contrib_successes}/{total_successes} ({low_contrib_successes/total_successes*100:.1f}%)"
        )

    # 仮説の検証
    print(f"\n## 5. 仮説の検証")
    print("-" * 40)

    print("### 仮説: 「貢献量の多い開発者しか当たっていない」")

    if top3_successes:
        # 全体の貢献量分布
        total_high = sum(count for _, count in high_contributors)
        total_medium = sum(count for _, count in medium_contributors)
        total_low = sum(count for _, count in low_contributors)

        total_available_tasks = total_high + total_medium + total_low

        high_ratio_overall = total_high / total_available_tasks * 100
        medium_ratio_overall = total_medium / total_available_tasks * 100
        low_ratio_overall = total_low / total_available_tasks * 100

        print(f"\n   全体の貢献量分布:")
        print(f"     高貢献者: {high_ratio_overall:.1f}%")
        print(f"     中貢献者: {medium_ratio_overall:.1f}%")
        print(f"     低貢献者: {low_ratio_overall:.1f}%")

        # 成功の貢献量分布（再掲）
        contributions = [s["contribution"] for s in top3_successes]
        high_success_ratio = (
            sum(1 for c in contributions if c >= 50) / len(contributions) * 100
        )
        medium_success_ratio = (
            sum(1 for c in contributions if 10 <= c < 50) / len(contributions) * 100
        )
        low_success_ratio = (
            sum(1 for c in contributions if c < 10) / len(contributions) * 100
        )

        print(f"\n   成功の貢献量分布:")
        print(f"     高貢献者: {high_success_ratio:.1f}%")
        print(f"     中貢献者: {medium_success_ratio:.1f}%")
        print(f"     低貢献者: {low_success_ratio:.1f}%")

        # バイアスの判定
        high_bias = high_success_ratio - high_ratio_overall
        medium_bias = medium_success_ratio - medium_ratio_overall
        low_bias = low_success_ratio - low_ratio_overall

        print(f"\n   バイアス分析（成功率 - 全体率）:")
        print(f"     高貢献者バイアス: {high_bias:+.1f}%ポイント")
        print(f"     中貢献者バイアス: {medium_bias:+.1f}%ポイント")
        print(f"     低貢献者バイアス: {low_bias:+.1f}%ポイント")

        # 結論
        if high_bias > 10:
            print(f"\n   🎯 結論: 仮説は正しい - 高貢献者に強いバイアス")
        elif abs(high_bias) < 5:
            print(f"\n   🎯 結論: 仮説は間違い - バイアスは軽微")
        else:
            print(f"\n   🎯 結論: 仮説は部分的に正しい - 中程度のバイアス")

    return {
        "successful_predictions": successful_predictions,
        "author_contribution": author_contribution,
        "high_contributors": high_contributors,
        "medium_contributors": medium_contributors,
        "low_contributors": low_contributors,
    }


if __name__ == "__main__":
    results = analyze_successful_predictions()

    print(f"\n🎯 まとめ:")
    top3_count = len(results["successful_predictions"]["top3"])
    total_contributors = len(results["author_contribution"])
    print(f"   Top-3成功数: {top3_count}")
    print(f"   成功した開発者の詳細リストを上記に表示")
    print(f"   貢献量バイアスの分析結果を確認")
