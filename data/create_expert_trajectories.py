import json
import pickle

from src.kazoo.consts.actions import Action


def map_event_to_action(event):
    """GitHubイベントをシミュレータのActionにマッピングする"""
    event_type = event.get("type")
    # このマッピングは研究の仮説に合わせて拡張・修正する必要がある
    if event_type == "issue_assigned":
        return Action.ASSIGN_TASK, {
            "issue_number": event["number"],
            "assignee": event.get("actor"),
        }
    if event_type == "pr_opened":
        return Action.SUBMIT_PULL_REQUEST, {
            "pr_number": event["number"],
            "author": event.get("actor"),
        }
    if event_type == "pr_review_approved":
        return Action.APPROVE_PULL_REQUEST, {
            "pr_number": event["number"],
            "reviewer": event.get("actor"),
        }
    if event_type == "pr_merged":
        return Action.MERGE_PULL_REQUEST, {
            "pr_number": event["number"],
            "merger": event.get("actor"),
        }
    return None, None


def main():
    with open("data/expert_events_detailed.json", "r") as f:
        events = json.load(f)

    trajectories = []
    current_trajectory = []

    # シミュレーション上の仮想的なプロジェクト状態。この状態の更新が重要。
    simulated_state = {"open_issues": {}, "open_prs": {}, "developer_status": {}}

    for event in events:
        # イベント発生「前」のプロジェクト状態を、今回のステップの状態とする
        expert_state = simulated_state.copy()

        action_enum, action_details = map_event_to_action(event)

        if action_enum and action_details and event.get("actor"):
            current_trajectory.append(
                {
                    "state": expert_state,
                    "action": action_enum,
                    "action_details": action_details,
                    "actor": event.get("actor"),
                }
            )

        # 【最重要】イベントに基づき、次のステップの`simulated_state`を更新するロジック
        # 例：PRがオープンされたら、simulated_state["open_prs"]にPR情報を追加する
        # 例：開発者がタスクにアサインされたら、developer_statusを更新する
        # この部分の精緻さが、研究の質を決定します。

        if event.get("type") in ["issue_closed", "pr_merged"]:
            if current_trajectory:
                trajectories.append(current_trajectory)
                current_trajectory = []

    output_path = "data/expert_trajectories.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(trajectories, f)
    print(
        f"Expert trajectories saved to {output_path}. Total trajectories: {len(trajectories)}"
    )


if __name__ == "__main__":
    main()
