#!/usr/bin/env python3
"""
ndeloofが推薦から除外された理由の詳細調査
最高貢献者が推薦されない原因を解明
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


def investigate_ndeloof_exclusion():
    """ndeloofが除外された理由を詳細調査"""
    print("🔍 ndeloof除外の詳細調査")
    print("=" * 60)

    # データ読み込み
    tasks, ground_truth = load_test_data_with_bot_filtering(
        "data/backlog_test_2023.json"
    )
    trained_models = load_sample_models(
        "models/improved_rl/final_models", ground_truth, 50
    )

    # 貢献量分析
    author_contribution = Counter(ground_truth)

    print(f"## 1. 基本情報確認")
    print("-" * 40)
    print(f"   ndeloofの貢献量: {author_contribution.get('ndeloof', 0)}タスク")
    print(
        f"   ndeloofが訓練済みモデルに含まれるか: {'✅' if 'ndeloof' in trained_models else '❌'}"
    )

    if "ndeloof" not in trained_models:
        print(f"\n🚨 重大な発見: ndeloofが訓練済みモデルに含まれていません！")

        # 利用可能なモデルを確認
        print(f"\n   利用可能なモデル一覧:")
        for i, agent in enumerate(sorted(trained_models.keys()), 1):
            contribution = author_contribution.get(agent, 0)
            print(f"     {i:2d}. {agent}: {contribution}タスク")

        # ndeloofのモデルファイルが存在するか確認
        model_dir = "models/improved_rl/final_models"
        ndeloof_model_path = os.path.join(model_dir, "agent_ndeloof.pth")

        print(f"\n   ndeloofのモデルファイル確認:")
        print(f"     パス: {ndeloof_model_path}")
        print(f"     存在: {'✅' if os.path.exists(ndeloof_model_path) else '❌'}")

        if os.path.exists(ndeloof_model_path):
            print(f"\n   🔍 モデルファイルは存在するが読み込みに失敗している可能性")

            # モデル読み込みを試行
            try:
                model_data = torch.load(
                    ndeloof_model_path, map_location="cpu", weights_only=False
                )
                print(f"     モデル読み込み: ✅ 成功")
                print(f"     モデルキー: {list(model_data.keys())}")

                # ポリシーネットワーク再構築を試行
                try:
                    policy_network = PPOPolicyNetwork()
                    policy_network.load_state_dict(model_data["policy_state_dict"])
                    policy_network.eval()
                    print(f"     ポリシーネットワーク構築: ✅ 成功")

                    # 手動でモデルを追加
                    trained_models["ndeloof"] = policy_network
                    print(f"     手動追加: ✅ 完了")

                except Exception as e:
                    print(f"     ポリシーネットワーク構築: ❌ 失敗 - {e}")

            except Exception as e:
                print(f"     モデル読み込み: ❌ 失敗 - {e}")

        else:
            print(f"\n   🚨 ndeloofのモデルファイルが存在しません")

            # 全モデルファイルを確認
            all_model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
            print(f"     総モデルファイル数: {len(all_model_files)}")

            # ndeloofに類似するファイル名を検索
            ndeloof_like = [f for f in all_model_files if "ndeloof" in f.lower()]
            print(f"     ndeloof類似ファイル: {ndeloof_like}")

    # ndeloofが含まれている場合の分析
    if "ndeloof" in trained_models:
        print(f"\n## 2. ndeloofのスコア分析")
        print("-" * 40)

        # ndeloofのタスクでのスコア分析
        ndeloof_tasks = [
            (i, task)
            for i, (task, author) in enumerate(zip(tasks, ground_truth))
            if author == "ndeloof"
        ]

        print(f"   ndeloofのタスク数: {len(ndeloof_tasks)}")

        if ndeloof_tasks:
            sample_tasks = ndeloof_tasks[:5]  # 最初の5つのタスクを分析

            for i, (task_idx, task) in enumerate(sample_tasks, 1):
                print(f"\n   タスク{i}: {task.get('title', '')[:50]}...")

                try:
                    task_features = extract_task_features_for_model(task)

                    # 全エージェントのスコアを計算
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

                    # ndeloofの順位を確認
                    ndeloof_rank = None
                    ndeloof_score = agent_scores.get("ndeloof", 0.0)

                    for rank, (agent, score) in enumerate(sorted_agents, 1):
                        if agent == "ndeloof":
                            ndeloof_rank = rank
                            break

                    print(f"     ndeloofのスコア: {ndeloof_score:.3f}")
                    print(f"     ndeloofの順位: {ndeloof_rank}/{len(sorted_agents)}")

                    # 上位5位を表示
                    print(f"     上位5位:")
                    for rank, (agent, score) in enumerate(sorted_agents[:5], 1):
                        marker = "👑" if agent == "ndeloof" else "  "
                        contribution = author_contribution.get(agent, 0)
                        print(
                            f"       {rank}. {marker} {agent}: {score:.3f} ({contribution}タスク)"
                        )

                    # 貢献量バランス推薦での結果
                    print(f"\n     貢献量バランス推薦結果:")

                    # 高貢献者カテゴリ
                    high_contributors = set()
                    medium_contributors = set()
                    low_contributors = set()

                    for author, count in author_contribution.items():
                        if author in trained_models:
                            if count >= 50:
                                high_contributors.add(author)
                            elif count >= 10:
                                medium_contributors.add(author)
                            else:
                                low_contributors.add(author)

                    print(
                        f"       ndeloofのカテゴリ: {'高貢献者' if 'ndeloof' in high_contributors else '中貢献者' if 'ndeloof' in medium_contributors else '低貢献者'}"
                    )

                    # 各カテゴリでの順位
                    high_candidates = [
                        (agent, score)
                        for agent, score in agent_scores.items()
                        if agent in high_contributors
                    ]
                    high_candidates.sort(key=lambda x: x[1], reverse=True)

                    ndeloof_high_rank = None
                    for rank, (agent, score) in enumerate(high_candidates, 1):
                        if agent == "ndeloof":
                            ndeloof_high_rank = rank
                            break

                    print(
                        f"       高貢献者内での順位: {ndeloof_high_rank}/{len(high_candidates)}"
                    )
                    print(
                        f"       高貢献者上位2位: {[agent for agent, _ in high_candidates[:2]]}"
                    )

                    # なぜ除外されたかの分析
                    if ndeloof_high_rank and ndeloof_high_rank > 2:
                        print(
                            f"       🎯 除外理由: 高貢献者内で{ndeloof_high_rank}位のため、上位2位に入らなかった"
                        )

                except Exception as e:
                    print(f"     エラー: {e}")

    # 根本原因の分析
    print(f"\n## 3. 根本原因の分析")
    print("-" * 40)

    if "ndeloof" not in trained_models:
        print("### 🚨 主要原因: ndeloofが訓練済みモデルに含まれていない")
        print("   考えられる理由:")
        print("   1. モデルファイルが存在しない")
        print("   2. モデル読み込み時にエラーが発生")
        print("   3. max_models制限により除外")
        print("   4. 重複チェックで除外")

        # max_models制限の確認
        print(f"\n   max_models制限の確認:")
        print(f"     現在の制限: 50モデル")
        print(
            f"     実際の重複開発者数: {len(set(ground_truth).intersection(set([f.replace('agent_', '').replace('.pth', '') for f in os.listdir('models/improved_rl/final_models') if f.endswith('.pth')])))}人"
        )

    else:
        print("### 🎯 主要原因: 高貢献者カテゴリ内での競争に敗北")
        print("   ndeloofは高貢献者だが、同カテゴリ内で上位2位に入れない")
        print("   他の高貢献者（milas, glours）がより高いスコアを獲得")

    # 解決策の提案
    print(f"\n## 4. 解決策の提案")
    print("-" * 40)

    if "ndeloof" not in trained_models:
        print("### 即座の解決策:")
        print("   1. max_models制限を増加（50 → 100）")
        print("   2. ndeloofのモデル読み込みエラーを修正")
        print("   3. 重要開発者の優先読み込み実装")

    else:
        print("### 推薦アルゴリズムの改善:")
        print("   1. 高貢献者枠を3人に増加（2 → 3）")
        print("   2. 貢献量重み付きスコア調整")
        print("   3. 最高貢献者の優先選出")

    print("\n### 長期的改善:")
    print("   1. 全エージェントモデルの使用（7,001 → 全て）")
    print("   2. 動的カテゴリ調整")
    print("   3. 個別重要度による重み付け")

    return {
        "ndeloof_in_models": "ndeloof" in trained_models,
        "ndeloof_contribution": author_contribution.get("ndeloof", 0),
        "total_models": len(trained_models),
        "available_agents": list(trained_models.keys()),
    }


if __name__ == "__main__":
    results = investigate_ndeloof_exclusion()

    print(f"\n🎯 調査結果まとめ:")
    print(
        f"   ndeloofがモデルに含まれる: {'✅' if results['ndeloof_in_models'] else '❌'}"
    )
    print(f"   ndeloofの貢献量: {results['ndeloof_contribution']}タスク")
    print(f"   利用可能モデル数: {results['total_models']}")

    if not results["ndeloof_in_models"]:
        print(f"   🚨 緊急対応が必要: 最高貢献者が推薦対象外")
