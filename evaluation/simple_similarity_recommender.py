#!/usr/bin/env python3
"""
シンプルな類似度ベースのタスク推薦システム

- 基本的なテキストベースの特徴量のみ使用
- TF-IDFとコサイン類似度で推薦
- 開発者の履歴ベースで類似度計算
- 時系列ベースの動的候補プール対応
"""

import argparse
import json
import pickle
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class SimpleSimilarityRecommender:
    """シンプルな類似度ベースのタスク推薦システム"""

    def __init__(self, config_path):
        """
        Args:
            config_path: 設定ファイルパス
        """
        self.config_path = config_path

        # 設定読み込み
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # TF-IDFベクトライザー
        self.vectorizer = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2)
        )

        # 学習済みモデル保存用
        self.trained_models = {}
        self.developer_profiles = {}
        self.scaler = StandardScaler()

        # 時系列ベースの開発者アクティビティデータ
        self.developer_activity_timeline = {}
        self.monthly_active_developers = defaultdict(set)

        print("🎯 シンプル類似度ベースの推薦システム初期化完了")

    def load_data(self, data_path):
        """データを読み込んで時系列分割"""
        print("📊 データ読み込み中...")

        with open(data_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)

        # 時系列分割
        training_data = []  # 2022年以前
        test_data = []  # 2023年

        for task in all_data:
            created_at = task.get("created_at", "")
            if created_at.startswith("2023"):
                test_data.append(task)
            elif created_at:  # 2022年以前の全データ
                year = int(created_at[:4])
                if year <= 2022:
                    training_data.append(task)
            elif created_at:  # 2022年以前の全データ
                year = int(created_at[:4])
                if year <= 2022:
                    training_data.append(task)

        print(f"   学習データ: {len(training_data):,} タスク (2014-2022年)")
        print(f"   テストデータ: {len(test_data):,} タスク (2023年)")

        return training_data, test_data

    def extract_developers_from_task(self, task_data, method="all"):
        """
        タスクから開発者を抽出（複数の方法に対応）

        Args:
            task_data: タスクデータ
            method: 抽出方法 ('assignees', 'creators', 'all')
                   - 'assignees': assigneesフィールドのみ（従来の方法）
                   - 'creators': Issue/PR作成者（userフィールド）も含む
                   - 'all': すべての方法を統合

        Returns:
            list: 抽出された開発者のリスト（priority順）
        """
        developers = []

        if method in ["assignees", "all"]:
            # 1. Assignees（最高優先度）
            if "assignees" in task_data and task_data["assignees"]:
                for assignee in task_data["assignees"]:
                    if "login" in assignee:
                        developers.append(
                            {
                                "login": assignee["login"],
                                "source": "assignees",
                                "priority": 1,
                            }
                        )

            # 2. Events (assigned) - 従来の補完方法
            elif "events" in task_data:
                for event in task_data["events"]:
                    if event.get("event") == "assigned" and event.get("assignee"):
                        login = event["assignee"].get("login")
                        if login:
                            developers.append(
                                {
                                    "login": login,
                                    "source": "events_assigned",
                                    "priority": 2,
                                }
                            )
                            break

        if method in ["creators", "all"]:
            # 3. Issue/PR作成者（新しい方法）
            if (
                "user" in task_data
                and task_data["user"]
                and "login" in task_data["user"]
            ):
                user_login = task_data["user"]["login"]
                # 既に他の方法で抽出されていない場合のみ追加
                existing_logins = {dev["login"] for dev in developers}
                if user_login not in existing_logins:
                    developers.append(
                        {"login": user_login, "source": "user_creator", "priority": 3}
                    )

        return developers

    def extract_training_pairs(self, training_data, extraction_method="all"):
        """
        学習データから開発者-タスクペアを抽出（複数の抽出方法に対応）

        Args:
            extraction_method: 'assignees' | 'creators' | 'all'
        """
        print(f"🔍 学習用ペア抽出中... (方法: {extraction_method})")

        training_pairs = []
        developer_stats = Counter()
        extraction_stats = Counter()  # 抽出方法の統計

        for task_data in training_data:
            task_id = task_data.get("id") or task_data.get("number")
            if not task_id:
                continue

            # 複数の方法で開発者を抽出
            developers = self.extract_developers_from_task(
                task_data, method=extraction_method
            )

            # 最高優先度の開発者を選択
            if developers:
                # 優先度でソート
                developers.sort(key=lambda x: x["priority"])
                selected_dev = developers[0]

                training_pairs.append(
                    {
                        "task_data": task_data,
                        "developer": selected_dev["login"],
                        "task_id": task_id,
                        "extraction_source": selected_dev["source"],
                    }
                )
                developer_stats[selected_dev["login"]] += 1
                extraction_stats[selected_dev["source"]] += 1

        print(f"   学習ペア: {len(training_pairs):,} ペア")
        print(f"   ユニーク開発者: {len(developer_stats)} 人")

        # 抽出方法の統計
        print("   抽出方法別統計:")
        for source, count in extraction_stats.most_common():
            print(f"     {source}: {count} ペア ({count/len(training_pairs)*100:.1f}%)")

        # 上位開発者表示
        top_devs = developer_stats.most_common(10)
        print("   上位開発者:")
        for dev, count in top_devs:
            print(f"     {dev}: {count} タスク")

        return training_pairs, developer_stats

    def build_developer_activity_timeline(self, training_data, extraction_method="all"):
        """
        学習データから開発者の時系列アクティビティを構築

        Args:
            extraction_method: 開発者抽出方法（'assignees', 'creators', 'all'）

        アクティブの定義:
        1. assignees: タスクに正式に割り当てられた開発者
        2. creators: Issue/PR作成者も含む
        3. all: すべての関与形態を統合
        """
        print(
            f"📅 開発者アクティビティタイムライン構築中... (方法: {extraction_method})"
        )

        # 月別開発者アクティビティ
        monthly_activity = defaultdict(lambda: defaultdict(int))
        total_assignments = 0
        extraction_stats = Counter()

        for task_data in training_data:
            created_at = task_data.get("created_at", "")
            if not created_at:
                continue

            # 月を抽出 (YYYY-MM)
            try:
                month = created_at[:7]
            except:
                continue

            # 新しい抽出方法を使用
            developers = self.extract_developers_from_task(
                task_data, method=extraction_method
            )

            if developers:
                # 最高優先度の開発者を選択
                developers.sort(key=lambda x: x["priority"])
                selected_dev = developers[0]

                assignee = selected_dev["login"]
                monthly_activity[month][assignee] += 1
                self.monthly_active_developers[month].add(assignee)
                total_assignments += 1
                extraction_stats[selected_dev["source"]] += 1

        # アクティビティタイムラインを保存
        self.developer_activity_timeline = dict(monthly_activity)

        print(
            f"   アクティビティタイムライン: {len(self.developer_activity_timeline)} ヶ月"
        )
        print(f"   総割り当て数: {total_assignments:,}")

        # 抽出方法の統計表示
        print("   抽出方法別統計:")
        for source, count in extraction_stats.most_common():
            print(f"     {source}: {count} タスク ({count/total_assignments*100:.1f}%)")

        # 月別統計表示（全期間）
        print("   月別アクティビティ:")
        for month in sorted(self.developer_activity_timeline.keys()):
            active_devs = len(self.monthly_active_developers[month])
            total_tasks = sum(self.developer_activity_timeline[month].values())
            print(f"     {month}: {active_devs} 開発者, {total_tasks} タスク")

        return self.developer_activity_timeline

    def get_active_developers_for_date(self, target_date, lookback_months=6):
        """
        指定日時から過去N ヶ月間でアクティブだった開発者リストを取得

        アクティブの定義:
        - タスクに割り当てられた（assigneeまたはassigned event）
        - 指定期間内に1つ以上のタスクを担当
        - 学習データ（2022年）での活動履歴がある
        """
        try:
            # 日時パース
            if isinstance(target_date, str):
                target_dt = datetime.fromisoformat(target_date.replace("Z", "+00:00"))
            else:
                target_dt = target_date

            # 検索範囲を計算（より柔軟な月計算）
            active_developers = set()

            for i in range(lookback_months + 1):  # 当月を含む
                # より正確な月計算
                if target_dt.month - i <= 0:
                    # 前年に遡る
                    year = target_dt.year - 1
                    month = 12 + (target_dt.month - i)
                else:
                    year = target_dt.year
                    month = target_dt.month - i

                search_month = f"{year:04d}-{month:02d}"

                if search_month in self.monthly_active_developers:
                    devs_in_month = self.monthly_active_developers[search_month]
                    active_developers.update(devs_in_month)
                    print(f"   {search_month}: {len(devs_in_month)} アクティブ開発者")

            return list(active_developers)

        except Exception as e:
            print(f"⚠️ 日時処理エラー ({target_date}): {e}")
            # フォールバック: 全開発者を返す
            all_devs = set()
            for month_devs in self.monthly_active_developers.values():
                all_devs.update(month_devs)
            return list(all_devs)

    def create_temporal_candidate_pool(self, task_data, learned_profiles):
        """テストタスクの作成日時ベースで動的候補プールを作成"""
        created_at = task_data.get("created_at", "")
        if not created_at:
            # 作成日時が不明な場合は全開発者を候補にする
            return learned_profiles

        # その時点でアクティブだった開発者を取得（直近6ヶ月）
        active_developers = self.get_active_developers_for_date(
            created_at, lookback_months=6
        )

        print(
            f"   📅 {created_at[:10]} 時点の直近6ヶ月アクティブ開発者: {len(active_developers)} 人"
        )

        # アクティブ開発者の中で学習済みプロファイルがある開発者
        temporal_pool = {}
        for dev_name in active_developers:
            if dev_name in learned_profiles:
                temporal_pool[dev_name] = learned_profiles[dev_name]
                temporal_pool[dev_name]["temporal_active"] = True
            else:
                # 学習済みプロファイルがない場合はデフォルトプロファイル作成
                temporal_pool[dev_name] = self._create_default_profile(dev_name)
                temporal_pool[dev_name]["temporal_active"] = True

        # 候補が少なすぎる場合は拡張（より長期間まで遡る）
        if len(temporal_pool) < 3:
            print(f"⚠️ 候補プールが小さすぎます ({len(temporal_pool)} 人), 拡張中...")
            # より長期間から追加で開発者を探す
            for i in range(7, 12):  # 7-11ヶ月前まで拡張（最大1年前）
                extra_devs = self.get_active_developers_for_date(
                    created_at, lookback_months=i
                )
                for dev_name in extra_devs:
                    if dev_name not in temporal_pool:
                        if dev_name in learned_profiles:
                            temporal_pool[dev_name] = learned_profiles[dev_name]
                            temporal_pool[dev_name][
                                "temporal_active"
                            ] = False  # 間接的にアクティブ
                        else:
                            temporal_pool[dev_name] = self._create_default_profile(
                                dev_name
                            )
                            temporal_pool[dev_name]["temporal_active"] = False

                if len(temporal_pool) >= 5:  # 最低5人を確保
                    print(f"   拡張完了: {len(temporal_pool)} 人 ({i}ヶ月前まで検索)")
                    break

        return temporal_pool

    def extract_task_text(self, task_data):
        """タスクからテキストを抽出"""
        # データ型チェック
        if isinstance(task_data, str):
            return task_data  # 文字列の場合はそのまま返す
        elif not isinstance(task_data, dict):
            return ""  # 辞書でない場合は空文字を返す

        text_parts = []

        # タイトル
        if "title" in task_data:
            text_parts.append(task_data["title"])

        # 本文
        body = task_data.get("body", "") or ""  # Noneの場合も空文字列に
        if body:
            # マークダウンやHTMLタグを簡単に除去
            body = re.sub(r"<[^>]*>", "", body)
            body = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
            body = re.sub(r"`[^`]*`", "", body)
            text_parts.append(body)

        # ラベル
        if "labels" in task_data:
            labels = [
                label.get("name", "") if isinstance(label, dict) else str(label)
                for label in task_data["labels"]
            ]
            text_parts.extend(labels)

        return " ".join(text_parts)

    def extract_basic_features(self, task_data):
        """基本的な特徴量を抽出"""
        # データ型チェック
        if not isinstance(task_data, dict):
            return {}  # 辞書でない場合は空の特徴量を返す

        features = {}

        # タスクの基本情報
        features["title_length"] = len(task_data.get("title", ""))
        body = task_data.get("body", "") or ""  # Noneの場合も空文字列に
        features["body_length"] = len(body)
        features["comments_count"] = task_data.get("comments", 0)

        # ラベルベースの特徴量
        labels = []
        if "labels" in task_data and task_data["labels"]:
            for label in task_data["labels"]:
                if isinstance(label, dict):
                    labels.append(label.get("name", "").lower())
                elif isinstance(label, str):
                    labels.append(label.lower())

        features["is_bug"] = 1 if any("bug" in label for label in labels) else 0
        features["is_enhancement"] = (
            1
            if any("enhancement" in label or "feature" in label for label in labels)
            else 0
        )
        features["is_documentation"] = (
            1 if any("doc" in label for label in labels) else 0
        )
        features["is_question"] = (
            1 if any("question" in label for label in labels) else 0
        )
        features["is_help_wanted"] = (
            1 if any("help" in label for label in labels) else 0
        )
        features["label_count"] = len(labels)

        # 状態
        features["is_open"] = 1 if task_data.get("state") == "open" else 0

        return features

    def build_developer_profiles(self, training_pairs):
        """開発者プロファイルを構築"""
        print("👥 開発者プロファイル構築中...")

        developer_profiles = defaultdict(
            lambda: {
                "task_count": 0,
                "task_texts": [],
                "feature_sums": defaultdict(float),
                "label_preferences": defaultdict(int),
            }
        )

        for pair in training_pairs:
            developer = pair["developer"]
            task_data = pair["task_data"]

            # テキスト収集
            task_text = self.extract_task_text(task_data)
            developer_profiles[developer]["task_texts"].append(task_text)

            # 基本特徴量集計
            features = self.extract_basic_features(task_data)
            for key, value in features.items():
                developer_profiles[developer]["feature_sums"][key] += value

            # ラベル嗜好
            labels = []
            if "labels" in task_data and task_data["labels"]:
                for label in task_data["labels"]:
                    if isinstance(label, dict):
                        labels.append(label.get("name", "").lower())
                    elif isinstance(label, str):
                        labels.append(label.lower())

            for label in labels:
                developer_profiles[developer]["label_preferences"][label] += 1

            developer_profiles[developer]["task_count"] += 1

        # 平均特徴量を計算
        for dev_name, profile in developer_profiles.items():
            if profile["task_count"] > 0:
                for key, total in profile["feature_sums"].items():
                    profile[f"avg_{key}"] = total / profile["task_count"]

        print(f"   開発者プロファイル完了: {len(developer_profiles)} 人")
        return dict(developer_profiles)

    def train_text_similarity_model(self, developer_profiles):
        """テキストベースの類似度モデルを訓練"""
        print("📝 テキスト類似度モデル訓練中...")

        # 各開発者の統合テキストを作成
        dev_texts = []
        dev_names = []

        for dev_name, profile in developer_profiles.items():
            if profile["task_texts"]:
                combined_text = " ".join(profile["task_texts"])
                dev_texts.append(combined_text)
                dev_names.append(dev_name)

        # TF-IDFベクトル化
        if dev_texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(dev_texts)
            self.dev_names = dev_names

            print(f"   TF-IDF行列: {self.tfidf_matrix.shape}")
            print(f"   語彙サイズ: {len(self.vectorizer.vocabulary_)}")
        else:
            print("⚠️ テキストデータが不足しています")

    def train_feature_model(self, training_pairs):
        """特徴量ベースのモデルを訓練"""
        print("🔧 特徴量モデル訓練中...")

        features = []
        labels = []

        for pair in training_pairs:
            task_features = self.extract_basic_features(pair["task_data"])
            feature_vector = [
                task_features[key] for key in sorted(task_features.keys())
            ]

            features.append(feature_vector)
            labels.append(pair["developer"])

        if features:
            features = np.array(features)

            # 正規化
            features_scaled = self.scaler.fit_transform(features)

            # RandomForest分類器
            self.rf_classifier = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )

            self.rf_classifier.fit(features_scaled, labels)

            print(f"   特徴量モデル訓練完了: {len(set(labels))} クラス")

            # 特徴重要度
            feature_names = sorted(
                self.extract_basic_features(training_pairs[0]["task_data"]).keys()
            )
            importances = self.rf_classifier.feature_importances_

            print("   特徴重要度 Top-5:")
            for i, (feat, imp) in enumerate(zip(feature_names, importances)):
                if i < 5:
                    print(f"     {feat}: {imp:.3f}")
        else:
            print("⚠️ 特徴量データが不足しています")

    def predict_assignments(self, test_data, developer_profiles):
        """テストデータの割り当てを予測"""
        print("🤖 割り当て予測中...")

        predictions = {}
        prediction_scores = {}

        # テストデータの実際の割り当てを抽出
        test_assignments = {}
        for task_data in test_data:
            task_id = task_data.get("id") or task_data.get("number")
            if not task_id:
                continue

            assignee = None
            if "assignees" in task_data and task_data["assignees"]:
                assignee = task_data["assignees"][0].get("login")
            elif "events" in task_data:
                for event in task_data["events"]:
                    if event.get("event") == "assigned" and event.get("assignee"):
                        assignee = event["assignee"].get("login")
                        break

            if assignee:
                test_assignments[task_id] = assignee

        print(f"   予測対象: {len(test_assignments)} タスク")

        prediction_count = 0

        # 各タスクの予測
        for task_data in test_data:
            task_id = task_data.get("id") or task_data.get("number")
            if task_id not in test_assignments:
                continue

            try:
                # テキスト類似度による予測
                text_scores = self._predict_by_text_similarity(task_data)

                # 特徴量類似度による予測
                feature_scores = self._predict_by_features(task_data)

                # 両方の予測を統合
                combined_scores = self._combine_predictions(text_scores, feature_scores)

                if combined_scores:
                    # スコア順にソート
                    sorted_scores = sorted(
                        combined_scores.items(), key=lambda x: x[1], reverse=True
                    )

                    # 最高スコアの開発者を予測
                    best_dev = sorted_scores[0][0]

                    predictions[task_id] = best_dev
                    prediction_scores[task_id] = {
                        "predicted_dev": best_dev,
                        "combined_score": sorted_scores[0][1],
                        "text_score": text_scores.get(best_dev, 0.0),
                        "feature_score": feature_scores.get(best_dev, 0.0),
                        "all_scores": dict(
                            sorted_scores
                        ),  # 全開発者のスコア（ソート済み）
                    }

                    prediction_count += 1

                    if prediction_count % 100 == 0:
                        print(f"   進捗: {prediction_count}/{len(test_assignments)}")

            except Exception as e:
                print(f"⚠️ タスク {task_id} の予測エラー: {e}")
                continue

        print(f"   予測完了: {len(predictions)} タスク")
        return predictions, prediction_scores, test_assignments

    def predict_assignments_with_pool(
        self, test_data, developer_profiles, test_assignments
    ):
        """テストベース候補プールでの割り当て予測"""
        print("🤖 テストベース割り当て予測中...")

        predictions = {}
        prediction_scores = {}

        print(f"   予測対象: {len(test_assignments)} タスク")
        print(f"   候補開発者: {len(developer_profiles)} 人")

        prediction_count = 0

        # 各タスクの予測
        for task_data in test_data:
            # データ型チェック
            if isinstance(task_data, str):
                print(f"⚠️ タスクデータが文字列形式: {task_data[:100]}...")
                continue
            elif not isinstance(task_data, dict):
                print(f"⚠️ タスクデータが辞書形式ではありません: {type(task_data)}")
                continue

            task_id = task_data.get("id") or task_data.get("number")
            if task_id not in test_assignments:
                continue

            try:
                # テキスト類似度による予測
                text_scores = self._predict_by_text_similarity(task_data)

                # 特徴量類似度による予測
                feature_scores = self._predict_by_features(task_data)

                # 両方の予測を統合
                combined_scores = self._combine_predictions(text_scores, feature_scores)

                if combined_scores:
                    # スコア順にソート
                    sorted_scores = sorted(
                        combined_scores.items(), key=lambda x: x[1], reverse=True
                    )

                    # 最高スコアの開発者を予測
                    best_dev = sorted_scores[0][0]

                    predictions[task_id] = best_dev
                    prediction_scores[task_id] = {
                        "predicted_dev": best_dev,
                        "combined_score": sorted_scores[0][1],
                        "text_score": text_scores.get(best_dev, 0.0),
                        "feature_score": feature_scores.get(best_dev, 0.0),
                        "all_scores": dict(
                            sorted_scores
                        ),  # 全開発者のスコア（ソート済み）
                    }

                    prediction_count += 1

                    if prediction_count % 100 == 0:
                        print(f"   進捗: {prediction_count}/{len(test_assignments)}")

            except Exception as e:
                print(f"⚠️ タスク {task_id} の予測エラー: {e}")
                continue

        print(f"   予測完了: {len(predictions)} タスク")
        return predictions, prediction_scores, test_assignments

    def predict_assignments_temporal(self, test_data, learned_profiles):
        """時系列ベース候補プールでの割り当て予測"""
        print("🕐 時系列ベース割り当て予測中...")

        predictions = {}
        prediction_scores = {}

        # テストデータの実際の割り当てを抽出
        test_assignments = {}
        for task_data in test_data:
            task_id = task_data.get("id") or task_data.get("number")
            if not task_id:
                continue

            assignee = None
            if "assignees" in task_data and task_data["assignees"]:
                assignee = task_data["assignees"][0].get("login")
            elif "events" in task_data:
                for event in task_data["events"]:
                    if event.get("event") == "assigned" and event.get("assignee"):
                        assignee = event["assignee"].get("login")
                        break

            if assignee:
                test_assignments[task_id] = assignee

        print(f"   予測対象: {len(test_assignments)} タスク")

        prediction_count = 0
        temporal_stats = {
            "avg_pool_size": 0,
            "min_pool_size": float("inf"),
            "max_pool_size": 0,
            "pool_sizes": [],
        }

        # 各タスクの予測
        for task_data in test_data:
            # データ型チェック
            if isinstance(task_data, str):
                print(f"⚠️ タスクデータが文字列形式: {task_data[:100]}...")
                continue
            elif not isinstance(task_data, dict):
                print(f"⚠️ タスクデータが辞書形式ではありません: {type(task_data)}")
                continue

            task_id = task_data.get("id") or task_data.get("number")
            if task_id not in test_assignments:
                continue

            try:
                # その時点の動的候補プールを作成
                temporal_pool = self.create_temporal_candidate_pool(
                    task_data, learned_profiles
                )

                # 統計収集
                pool_size = len(temporal_pool)
                temporal_stats["pool_sizes"].append(pool_size)
                temporal_stats["min_pool_size"] = min(
                    temporal_stats["min_pool_size"], pool_size
                )
                temporal_stats["max_pool_size"] = max(
                    temporal_stats["max_pool_size"], pool_size
                )

                # 一時的に開発者プロファイルを設定
                original_profiles = self.developer_profiles
                self.developer_profiles = temporal_pool

                # テキスト類似度による予測
                text_scores = self._predict_by_text_similarity(task_data)

                # 特徴量類似度による予測
                feature_scores = self._predict_by_features(task_data)

                # 元のプロファイルを復元
                self.developer_profiles = original_profiles

                # 両方の予測を統合
                combined_scores = self._combine_predictions(text_scores, feature_scores)

                # 時系列アクティビティによる重み付け
                combined_scores = self._apply_temporal_weights(
                    combined_scores, temporal_pool, task_data
                )

                if combined_scores:
                    # スコア順にソート
                    sorted_scores = sorted(
                        combined_scores.items(), key=lambda x: x[1], reverse=True
                    )

                    # 最高スコアの開発者を予測
                    best_dev = sorted_scores[0][0]

                    predictions[task_id] = best_dev
                    prediction_scores[task_id] = {
                        "predicted_dev": best_dev,
                        "combined_score": sorted_scores[0][1],
                        "text_score": text_scores.get(best_dev, 0.0),
                        "feature_score": feature_scores.get(best_dev, 0.0),
                        "all_scores": dict(sorted_scores),
                        "pool_size": pool_size,
                        "temporal_active": temporal_pool.get(best_dev, {}).get(
                            "temporal_active", False
                        ),
                        "created_at": task_data.get("created_at", ""),
                        "candidate_pool": list(temporal_pool.keys()),
                    }

                    prediction_count += 1

                    if prediction_count % 100 == 0:
                        print(f"   進捗: {prediction_count}/{len(test_assignments)}")

            except Exception as e:
                print(f"⚠️ タスク {task_id} の予測エラー: {e}")
                continue

        # 統計計算
        if temporal_stats["pool_sizes"]:
            temporal_stats["avg_pool_size"] = np.mean(temporal_stats["pool_sizes"])

        print(f"   予測完了: {len(predictions)} タスク")
        print(f"   平均候補プールサイズ: {temporal_stats['avg_pool_size']:.1f}")
        print(
            f"   候補プールサイズ範囲: {temporal_stats['min_pool_size']}-{temporal_stats['max_pool_size']}"
        )

        return predictions, prediction_scores, test_assignments, temporal_stats

    def _apply_temporal_weights(self, combined_scores, temporal_pool, task_data):
        """時系列アクティビティによる重み付けを適用"""
        weighted_scores = {}

        for dev_name, score in combined_scores.items():
            base_score = score

            # 時系列アクティビティによる重み
            if dev_name in temporal_pool:
                dev_profile = temporal_pool[dev_name]

                # 直近アクティブな開発者にボーナス
                if dev_profile.get("temporal_active", False):
                    base_score *= 1.2

                # アクティビティレベルによる調整
                activity_level = dev_profile.get("task_count", 1)
                activity_boost = min(1.3, 1.0 + (activity_level / 50))
                base_score *= activity_boost

            weighted_scores[dev_name] = base_score

        return weighted_scores

    def _predict_by_text_similarity(self, task_data):
        """テキスト類似度による予測"""
        scores = {}

        if hasattr(self, "tfidf_matrix") and hasattr(self, "dev_names"):
            task_text = self.extract_task_text(task_data)

            if task_text.strip():
                # タスクのTF-IDFベクトル
                task_vector = self.vectorizer.transform([task_text])

                # 類似度計算
                similarities = cosine_similarity(
                    task_vector, self.tfidf_matrix
                ).flatten()

                for i, dev_name in enumerate(self.dev_names):
                    scores[dev_name] = similarities[i]

        return scores

    def _predict_by_features(self, task_data):
        """特徴量による予測（拡張候補プール対応）"""
        scores = {}

        # タスクの特徴量を抽出
        task_features = self.extract_basic_features(task_data)

        # 全候補開発者に対してスコア計算
        for dev_name, profile in self.developer_profiles.items():
            if profile.get("source") == "training":
                # 学習済み開発者: RandomForest分類器を使用
                if hasattr(self, "rf_classifier"):
                    feature_vector = [
                        task_features[key] for key in sorted(task_features.keys())
                    ]
                    feature_vector = np.array(feature_vector).reshape(1, -1)
                    feature_vector_scaled = self.scaler.transform(feature_vector)

                    # 予測確率
                    probas = self.rf_classifier.predict_proba(feature_vector_scaled)[0]
                    classes = self.rf_classifier.classes_

                    for i, class_name in enumerate(classes):
                        if class_name == dev_name:
                            scores[dev_name] = probas[i]
                            break
                    else:
                        scores[dev_name] = 0.0
            else:
                # 既存プロファイル開発者: 特徴量類似度ベースの予測
                similarity_score = self._calculate_feature_similarity(
                    task_features, profile
                )
                scores[dev_name] = similarity_score

        return scores

    def _calculate_feature_similarity(self, task_features, dev_profile):
        """タスク特徴量と開発者プロファイルの類似度を計算"""
        # 基本的な特徴量類似度計算
        score = 0.0

        # タスクの長さ特徴量との類似度
        title_len_diff = abs(
            task_features["title_length"] - dev_profile.get("avg_title_length", 50)
        )
        body_len_diff = abs(
            task_features["body_length"] - dev_profile.get("avg_body_length", 200)
        )
        comments_diff = abs(
            task_features["comments_count"] - dev_profile.get("avg_comments_count", 2)
        )

        # 正規化して類似度に変換（差が小さいほど高スコア）
        title_sim = max(0, 1 - title_len_diff / 100)
        body_sim = max(0, 1 - body_len_diff / 1000)
        comments_sim = max(0, 1 - comments_diff / 10)

        # ラベル親和性
        label_affinity = 0.0
        if task_features["is_bug"]:
            label_affinity += dev_profile.get("avg_is_bug", 0.1)
        if task_features["is_enhancement"]:
            label_affinity += dev_profile.get("avg_is_enhancement", 0.1)
        if task_features["is_documentation"]:
            label_affinity += dev_profile.get("avg_is_documentation", 0.05)

        # 重み付き統合スコア
        score = (
            0.2 * title_sim + 0.3 * body_sim + 0.2 * comments_sim + 0.3 * label_affinity
        )

        # アクティビティレベルによる調整
        activity_boost = min(1.0, dev_profile.get("task_count", 1) / 10)
        score *= 0.5 + 0.5 * activity_boost

        return score

    def _combine_predictions(self, text_scores, feature_scores):
        """予測の統合"""
        combined_scores = {}

        # 重み設定
        text_weight = 0.6
        feature_weight = 0.4

        # 全開発者のリスト
        all_devs = set(text_scores.keys()) | set(feature_scores.keys())

        for dev in all_devs:
            text_score = text_scores.get(dev, 0.0)
            feature_score = feature_scores.get(dev, 0.0)

            combined_scores[dev] = (
                text_weight * text_score + feature_weight * feature_score
            )

        return combined_scores

    def evaluate_predictions(self, predictions, test_assignments, prediction_scores):
        """予測結果の評価"""
        print("📊 予測評価中...")

        # 共通タスクで評価
        common_tasks = set(predictions.keys()) & set(test_assignments.keys())
        print(f"   評価対象: {len(common_tasks)} タスク")

        if not common_tasks:
            print("⚠️ 評価可能なタスクがありません")
            return {}

        # Top-1 (従来の)正確性評価
        correct_predictions = 0
        for task_id in common_tasks:
            if predictions[task_id] == test_assignments[task_id]:
                correct_predictions += 1

        accuracy = correct_predictions / len(common_tasks)

        # Top-K評価 (K=3, 5)
        topk_metrics = self._evaluate_topk_accuracy(
            common_tasks, test_assignments, prediction_scores
        )

        # 信頼度統計
        prediction_confidences = []
        text_scores = []
        feature_scores = []

        for task_id in common_tasks:
            if task_id in prediction_scores:
                scores = prediction_scores[task_id]
                prediction_confidences.append(scores["combined_score"])
                text_scores.append(scores["text_score"])
                feature_scores.append(scores["feature_score"])

        metrics = {
            "accuracy": accuracy,
            "top1_accuracy": accuracy,  # 同じ値
            "top3_accuracy": topk_metrics["top3_accuracy"],
            "top5_accuracy": topk_metrics["top5_accuracy"],
            "correct_predictions": correct_predictions,
            "total_predictions": len(common_tasks),
            "avg_combined_score": (
                np.mean(prediction_confidences) if prediction_confidences else 0.0
            ),
            "avg_text_score": np.mean(text_scores) if text_scores else 0.0,
            "avg_feature_score": np.mean(feature_scores) if feature_scores else 0.0,
            "score_std": (
                np.std(prediction_confidences) if prediction_confidences else 0.0
            ),
        }

        print(
            f"   Top-1精度: {accuracy:.3f} ({correct_predictions}/{len(common_tasks)})"
        )
        print(
            f"   Top-3精度: {topk_metrics['top3_accuracy']:.3f} ({topk_metrics['top3_correct']}/{len(common_tasks)})"
        )
        print(
            f"   Top-5精度: {topk_metrics['top5_accuracy']:.3f} ({topk_metrics['top5_correct']}/{len(common_tasks)})"
        )
        print(
            f"   平均統合スコア: {metrics['avg_combined_score']:.3f} ± {metrics['score_std']:.3f}"
        )
        print(f"   平均テキストスコア: {metrics['avg_text_score']:.3f}")
        print(f"   平均特徴量スコア: {metrics['avg_feature_score']:.3f}")

        # 開発者別評価
        dev_metrics = self._evaluate_by_developer(
            predictions, test_assignments, common_tasks
        )
        metrics["developer_metrics"] = dev_metrics

        return metrics

    def _evaluate_topk_accuracy(
        self, common_tasks, test_assignments, prediction_scores
    ):
        """Top-K accuracy評価"""
        top3_correct = 0
        top5_correct = 0

        for task_id in common_tasks:
            actual_dev = test_assignments[task_id]

            if task_id in prediction_scores:
                # 全スコアを取得してソート
                all_scores = prediction_scores[task_id].get("all_scores", {})
                if not all_scores:
                    # all_scoresがない場合、combined_scoresを再構築
                    continue

                # スコア順に開発者を取得
                sorted_devs = list(all_scores.keys())

                # Top-3評価
                if len(sorted_devs) >= 3:
                    top3_devs = sorted_devs[:3]
                    if actual_dev in top3_devs:
                        top3_correct += 1
                elif actual_dev in sorted_devs:
                    top3_correct += 1

                # Top-5評価
                if len(sorted_devs) >= 5:
                    top5_devs = sorted_devs[:5]
                    if actual_dev in top5_devs:
                        top5_correct += 1
                elif actual_dev in sorted_devs:
                    top5_correct += 1

        total_tasks = len(common_tasks)
        return {
            "top3_accuracy": top3_correct / total_tasks if total_tasks > 0 else 0.0,
            "top5_accuracy": top5_correct / total_tasks if total_tasks > 0 else 0.0,
            "top3_correct": top3_correct,
            "top5_correct": top5_correct,
        }

    def _evaluate_by_developer(self, predictions, test_assignments, common_tasks):
        """開発者別の評価"""
        dev_metrics = defaultdict(lambda: {"correct": 0, "total": 0})

        for task_id in common_tasks:
            actual_dev = test_assignments[task_id]
            predicted_dev = predictions[task_id]

            dev_metrics[actual_dev]["total"] += 1
            if actual_dev == predicted_dev:
                dev_metrics[actual_dev]["correct"] += 1

        # 精度計算
        for dev, stats in dev_metrics.items():
            stats["accuracy"] = (
                stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            )

        print("   開発者別精度:")
        sorted_devs = sorted(
            dev_metrics.items(), key=lambda x: x[1]["total"], reverse=True
        )
        for dev, stats in sorted_devs[:10]:
            print(
                f"     {dev}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})"
            )

        return dict(dev_metrics)

    def save_model(self, output_dir="models"):
        """モデルの保存"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # モデル保存
        model_path = output_dir / f"simple_similarity_recommender_{timestamp}.pkl"
        with open(model_path, "wb") as f:
            model_data = {
                "vectorizer": self.vectorizer,
                "rf_classifier": getattr(self, "rf_classifier", None),
                "scaler": self.scaler,
                "developer_profiles": self.developer_profiles,
                "tfidf_matrix": getattr(self, "tfidf_matrix", None),
                "dev_names": getattr(self, "dev_names", []),
            }
            pickle.dump(model_data, f)

        print(f"✅ モデル保存: {model_path}")
        return model_path

    def run_full_pipeline(
        self,
        data_path,
        output_dir="outputs",
        use_temporal=True,
        extraction_method="all",
    ):
        """
        完全パイプラインの実行

        Args:
            extraction_method: 開発者抽出方法 ('assignees', 'creators', 'all')
        """
        print("🚀 シンプル類似度ベース推薦システム実行開始")
        if use_temporal:
            print("⏰ 時系列ベース動的候補プール有効")
        print(f"👥 開発者抽出方法: {extraction_method}")
        print("=" * 70)

        # 1. データ読み込み
        training_data, test_data = self.load_data(data_path)

        # 2. 学習ペア抽出（新しい抽出方法を使用）
        training_pairs, developer_stats = self.extract_training_pairs(
            training_data, extraction_method
        )

        if not training_pairs:
            print("⚠️ 学習ペアが見つかりませんでした")
            return {}

        # 3. 基本開発者プロファイル構築
        learned_profiles = self.build_developer_profiles(training_pairs)

        # 4. 時系列アクティビティタイムライン構築（新しい抽出方法を使用）
        if use_temporal:
            self.build_developer_activity_timeline(training_data, extraction_method)

        # 5. テキスト類似度モデル訓練（学習済みプロファイルのみで）
        self.train_text_similarity_model(learned_profiles)

        # 6. 特徴量モデル訓練（学習済みペアのみで）
        self.train_feature_model(training_pairs)

        # 7. 予測実行
        if use_temporal:
            # 時系列ベース動的候補プール
            predictions, prediction_scores, test_assignments, temporal_stats = (
                self.predict_assignments_temporal(test_data, learned_profiles)
            )
        else:
            # 従来のテストベース候補プール
            test_developers, test_assignments = self.extract_test_developers(test_data)
            self.developer_profiles = self.create_test_based_candidate_pool(
                learned_profiles, test_developers
            )
            predictions, prediction_scores, _ = self.predict_assignments_with_pool(
                test_data, self.developer_profiles, test_assignments
            )
            temporal_stats = {}

        # 8. 評価
        metrics = self.evaluate_predictions(
            predictions, test_assignments, prediction_scores
        )

        # 時系列統計を追加
        if use_temporal:
            metrics["temporal_stats"] = temporal_stats

        # 9. 結果保存
        self.save_results(
            metrics, predictions, prediction_scores, test_assignments, output_dir
        )

        # 10. モデル保存
        model_path = self.save_model()

        print("✅ シンプル類似度ベース推薦システム完了")
        return metrics

    def save_results(
        self, metrics, predictions, prediction_scores, test_assignments, output_dir
    ):
        """結果の保存"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # メトリクス保存
        metrics_path = output_dir / f"simple_similarity_metrics_{timestamp}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            # numpy配列を対応
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                return obj

            json.dump(metrics, f, indent=2, ensure_ascii=False, default=convert_numpy)

        # 予測結果保存
        results = []
        for task_id in set(predictions.keys()) | set(test_assignments.keys()):
            result = {
                "task_id": task_id,
                "actual_developer": test_assignments.get(task_id),
                "predicted_developer": predictions.get(task_id),
                "correct": test_assignments.get(task_id) == predictions.get(task_id),
            }

            if task_id in prediction_scores:
                result.update(prediction_scores[task_id])

            results.append(result)

        results_df = pd.DataFrame(results)
        results_path = output_dir / f"simple_similarity_results_{timestamp}.csv"
        results_df.to_csv(results_path, index=False, encoding="utf-8")

        print(f"✅ メトリクス保存: {metrics_path}")
        print(f"✅ 結果保存: {results_path}")

    def load_existing_profiles(self, profile_path="data/dev_profiles.json"):
        """既存の開発者プロファイルを読み込んで候補を拡張"""
        print("👥 既存開発者プロファイル読み込み中...")

        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                existing_profiles = json.load(f)

            # プロファイルを処理してSimpleSimilarityに適用可能な形式に変換
            expanded_profiles = {}

            for dev_name, profile in existing_profiles.items():
                # 基本的な統計情報を抽出
                if (
                    profile.get("total_contributions", 0) > 0
                ):  # 最低1つは貢献している開発者
                    expanded_profiles[dev_name] = {
                        "task_count": profile.get("total_contributions", 0),
                        "task_texts": [],  # 実際のタスクテキストは学習データから
                        "feature_sums": defaultdict(float),
                        "label_preferences": defaultdict(int),
                        "avg_title_length": profile.get("avg_issue_title_length", 0),
                        "avg_body_length": profile.get("avg_issue_body_length", 0),
                        "avg_comments_count": profile.get("avg_comments", 0),
                        "avg_is_bug": profile.get("label_affinities", {}).get(
                            "bug", 0.1
                        ),
                        "avg_is_enhancement": profile.get("label_affinities", {}).get(
                            "enhancement", 0.1
                        ),
                        "avg_is_documentation": profile.get("label_affinities", {}).get(
                            "documentation", 0.05
                        ),
                        "recent_activity": profile.get("recent_activity_score", 0),
                    }

            print(f"   既存プロファイル: {len(expanded_profiles)} 人")
            return expanded_profiles

        except FileNotFoundError:
            print(f"⚠️ プロファイルファイルが見つかりません: {profile_path}")
            # フォールバック: より多くの候補を生成
            return self._generate_fallback_profiles()
        except Exception as e:
            print(f"⚠️ プロファイル読み込みエラー: {e}")
            return self._generate_fallback_profiles()

    def _generate_fallback_profiles(self):
        """フォールバック: より多くの仮想開発者プロファイルを生成"""
        print("🔄 フォールバック候補プール生成中...")

        # よくある開発者名パターンを生成
        common_names = [
            "alice",
            "bob",
            "charlie",
            "david",
            "eve",
            "frank",
            "grace",
            "henry",
            "ivan",
            "julia",
            "kevin",
            "lisa",
            "mike",
            "nancy",
            "oscar",
            "penny",
            "quinn",
            "robert",
            "sara",
            "tom",
            "ursula",
            "victor",
            "wendy",
            "xavier",
            "yuki",
            "zoe",
            "alex",
            "jordan",
            "taylor",
            "casey",
            "jamie",
            "riley",
        ]

        fallback_profiles = {}
        for i, name in enumerate(common_names):
            fallback_profiles[f"dev_{name}"] = {
                "task_count": max(1, 20 - i),  # 異なるアクティビティレベル
                "task_texts": [],
                "feature_sums": defaultdict(float),
                "label_preferences": defaultdict(int),
                "avg_title_length": 40 + (i % 20),
                "avg_body_length": 150 + (i % 100),
                "avg_comments_count": 1 + (i % 5),
                "avg_is_bug": 0.05 + (i % 3) * 0.1,
                "avg_is_enhancement": 0.05 + ((i + 1) % 3) * 0.1,
                "avg_is_documentation": 0.02 + ((i + 2) % 3) * 0.05,
                "recent_activity": max(0.1, 1.0 - i * 0.03),
            }

        print(f"   フォールバック候補: {len(fallback_profiles)} 人")
        return fallback_profiles

    def expand_candidate_pool(
        self, learned_profiles, existing_profiles, min_activity_threshold=1
    ):
        """学習済みプロファイルと既存プロファイルを組み合わせて候補プールを拡張"""
        print("🔍 候補プール拡張中...")

        expanded_pool = {}

        # 1. 学習済みプロファイル（高品質）を優先
        for dev_name, profile in learned_profiles.items():
            expanded_pool[dev_name] = profile
            expanded_pool[dev_name]["source"] = "training"
            expanded_pool[dev_name]["priority"] = "high"

        # 2. 既存プロファイルから追加候補を選定
        added_from_existing = 0
        for dev_name, profile in existing_profiles.items():
            if (
                dev_name not in expanded_pool
                and profile.get("task_count", 0) >= min_activity_threshold
            ):
                # 既存プロファイルを学習済み形式に合わせて調整
                expanded_pool[dev_name] = {
                    "task_count": profile["task_count"],
                    "task_texts": [],  # 空のリスト（テキスト類似度では使用されない）
                    "feature_sums": profile.get("feature_sums", defaultdict(float)),
                    "label_preferences": profile.get(
                        "label_preferences", defaultdict(int)
                    ),
                    "avg_title_length": profile.get("avg_title_length", 50),
                    "avg_body_length": profile.get("avg_body_length", 200),
                    "avg_comments_count": profile.get("avg_comments_count", 2),
                    "avg_is_bug": profile.get("is_bug", 0.1),
                    "avg_is_enhancement": profile.get("is_enhancement", 0.1),
                    "avg_is_documentation": profile.get("is_documentation", 0.05),
                    "source": "existing",
                    "priority": "medium",
                }
                added_from_existing += 1

        print(f"   拡張候補プール: {len(expanded_pool)} 人")
        print(f"     - 学習済み: {len(learned_profiles)} 人")
        print(f"     - 既存から追加: {added_from_existing} 人")

        return expanded_pool

    def extract_test_developers(self, test_data):
        """テストデータから実際の担当開発者を抽出"""
        print("👥 テストデータ開発者抽出中...")

        test_developers = set()
        test_assignments = {}

        for task_data in test_data:
            task_id = task_data.get("id") or task_data.get("number")
            if not task_id:
                continue

            assignee = None
            if "assignees" in task_data and task_data["assignees"]:
                assignee = task_data["assignees"][0].get("login")
            elif "events" in task_data:
                for event in task_data["events"]:
                    if event.get("event") == "assigned" and event.get("assignee"):
                        assignee = event["assignee"].get("login")
                        break

            if assignee:
                test_developers.add(assignee)
                test_assignments[task_id] = assignee

        print(f"   テストデータ開発者: {len(test_developers)} 人")

        # 上位開発者表示
        dev_task_counts = Counter()
        for assignee in test_assignments.values():
            dev_task_counts[assignee] += 1

        top_test_devs = dev_task_counts.most_common(10)
        print("   テストデータ上位開発者:")
        for dev, count in top_test_devs:
            print(f"     {dev}: {count} タスク")

        return list(test_developers), test_assignments

    def create_test_based_candidate_pool(self, learned_profiles, test_developers):
        """テストデータ開発者ベースの候補プールを作成"""
        print("🎯 テストベース候補プール作成中...")

        candidate_pool = {}

        # 学習済みプロファイルがある開発者
        learned_count = 0
        for dev_name in test_developers:
            if dev_name in learned_profiles:
                candidate_pool[dev_name] = learned_profiles[dev_name]
                candidate_pool[dev_name]["source"] = "training"
                candidate_pool[dev_name]["priority"] = "high"
                learned_count += 1

        # 学習済みプロファイルがない開発者用のデフォルトプロファイル
        default_count = 0
        for dev_name in test_developers:
            if dev_name not in candidate_pool:
                candidate_pool[dev_name] = self._create_default_profile(dev_name)
                default_count += 1

        print(f"   候補プール: {len(candidate_pool)} 人")
        print(f"     - 学習済みプロファイル: {learned_count} 人")
        print(f"     - デフォルトプロファイル: {default_count} 人")

        return candidate_pool

    def _create_default_profile(self, dev_name):
        """学習データにない開発者用のデフォルトプロファイルを作成"""
        # 開発者名の特性に基づいてプロファイルを調整
        name_hash = hash(dev_name) % 100

        return {
            "task_count": 5 + (name_hash % 10),  # 5-14のランダムなタスク数
            "task_texts": [],
            "feature_sums": defaultdict(float),
            "label_preferences": defaultdict(int),
            "avg_title_length": 45 + (name_hash % 15),  # 45-59の平均タイトル長
            "avg_body_length": 180 + (name_hash % 40),  # 180-219の平均本文長
            "avg_comments_count": 2 + (name_hash % 3),  # 2-4の平均コメント数
            "avg_is_bug": 0.1 + (name_hash % 3) * 0.05,  # 0.1-0.2のバグ親和性
            "avg_is_enhancement": 0.1
            + ((name_hash + 1) % 3) * 0.05,  # 0.1-0.2の機能親和性
            "avg_is_documentation": 0.05
            + ((name_hash + 2) % 3) * 0.025,  # 0.05-0.1のドキュメント親和性
            "source": "default",
            "priority": "low",
        }


def main():
    parser = argparse.ArgumentParser(
        description="シンプル類似度ベースのタスク推薦システム"
    )
    parser.add_argument(
        "--config", default="configs/unified_rl.yaml", help="設定ファイルパス"
    )
    parser.add_argument("--data", default="data/backlog.json", help="統合データパス")
    parser.add_argument("--output", default="outputs", help="出力ディレクトリ")
    parser.add_argument(
        "--temporal", action="store_true", help="時系列ベース動的候補プール使用"
    )
    parser.add_argument(
        "--no-temporal",
        dest="temporal",
        action="store_false",
        help="従来のテストベース候補プール使用",
    )
    parser.add_argument(
        "--extraction-method",
        default="all",
        choices=["assignees", "creators", "all"],
        help="開発者抽出方法 (assignees: 割り当てのみ, creators: 作成者も含む, all: すべて)",
    )
    parser.set_defaults(temporal=True)

    args = parser.parse_args()

    # 推薦システム実行
    recommender = SimpleSimilarityRecommender(args.config)
    metrics = recommender.run_full_pipeline(
        args.data,
        args.output,
        use_temporal=args.temporal,
        extraction_method=args.extraction_method,
    )

    print("\n🎯 最終結果:")
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key not in [
                "developer_metrics",
                "temporal_stats",
            ]:
                print(f"   {key}: {value:.3f}")

        # 時系列統計表示
        if "temporal_stats" in metrics:
            stats = metrics["temporal_stats"]
            print(f"\n⏰ 時系列統計:")
            print(f"   平均候補プールサイズ: {stats.get('avg_pool_size', 0):.1f}")
            print(
                f"   候補プールサイズ範囲: {stats.get('min_pool_size', 0)}-{stats.get('max_pool_size', 0)}"
            )

    return 0


if __name__ == "__main__":
    exit(main())
