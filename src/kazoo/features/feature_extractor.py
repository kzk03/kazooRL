import numpy as np

from kazoo.consts.actions import Action

# 特徴量の次元数。増減させたら、下の関数とtrain_irl.pyの定義も変更する。
FEATURE_DIM = 3
FEATURE_NAMES = [
    "Task Completion(dummy)",  # TODO: タスク完了への貢献度を実装する
    "PR Size (dummy)",  # TODO: PRのサイズを実装する
    "Skill Match (dummy)",
    # "Review Cost",
    # "Action Cost",
]


def get_features(state, action_enum, action_details, actor_id):
    """状態と行動から特徴量ベクトルを計算する"""
    features = np.zeros(FEATURE_DIM)

    # 特徴量0: タスク完了への貢献度 (正の報酬が期待される)
    if action_enum in [
        Action.APPROVE_PULL_REQUEST,
        Action.MERGE_PULL_REQUEST,
        Action.CLOSE_ISSUE,
    ]:
        features[0] = 1.0

    # 特徴量1: PRのサイズ - 大きいPRはインセンティブかコストか？
    if action_enum == Action.SUBMIT_PULL_REQUEST:
        # TODO: stateからPRの情報を取得し、追加行数を特徴量とする
        # features[1] = get_pr_additions_from_state(state, action_details)
        pass

    # 特徴量2: スキルマッチ度 - 得意な作業か？ (正の報酬が期待される)
    # TODO: 開発者のスキルとタスクの要求スキルを比較するロジック
    # features[2] = calculate_skill_match(state, actor_id, action_details)
    features[2] = np.random.rand()  # 現状はダミー

    # # 特徴量3: レビュー負荷 - レビュー行為はコストか？ (負の報酬が期待される)
    # if action_enum == Action.APPROVE_PULL_REQUEST:
    #     features[3] = 1.0  # コストは学習される重みが負になることで表現

    # # 特徴量4: 行動あたりの基本コスト (負の報酬が期待される)
    # if action_enum != Action.DO_NOTHING:
    #     features[4] = 1.0

    return features
