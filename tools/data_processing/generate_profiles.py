import json
from collections import defaultdict

import yaml


def generate_developer_profiles(github_data_path, output_path):
    """
    生データから開発者のプロフィール（専門性、編集経験）を生成し、YAMLファイルに出力する。
    """
    with open(github_data_path, 'r') as f:
        data = json.load(f)

    # 開発者ごとの情報を集約する辞書
    # defaultdictを使うとキーが存在しない場合に自動で初期化してくれるので便利
    dev_stats = defaultdict(lambda: {
        'label_counts': defaultdict(int),
        'touched_files': set()
    })

    # 生データをループして情報を集計
    for event in data:
        # PRがマージされたイベントを対象とする
        if event.get('type') == 'PullRequestEvent' and event['payload'].get('action') == 'closed' and event['payload']['pull_request'].get('merged'):
            pr = event['payload']['pull_request']
            developer = pr['user']['login']
            
            # ラベルのカウント
            for label in pr.get('labels', []):
                dev_stats[developer]['label_counts'][label['name']] += 1
            
            # 編集ファイルの追加 (このデータが取得できている前提)
            # changed_filesは例です。実際のキーはデータ構造に合わせてください。
            # この情報を取得するにはget_github_data.pyでPRのファイルリストを取得する必要があります。
            for file_info in pr.get('files', []): # 仮に 'files' というキーにファイルリストが入っているとする
                dev_stats[developer]['touched_files'].add(file_info['filename'])

    # 集計結果を最終的なプロフィール形式に変換
    final_profiles = {}
    for dev, stats in dev_stats.items():
        total_labels = sum(stats['label_counts'].values())
        
        # ラベルの親和性（割合）を計算
        label_affinity = {
            label: count / total_labels for label, count in stats['label_counts'].items()
        } if total_labels > 0 else {}
        
        final_profiles[dev] = {
            'skills': [], # スキルは別途手動で定義することを想定
            'label_affinity': label_affinity,
            'touched_files': sorted(list(stats['touched_files'])) # setをリストに変換
        }

    # YAMLファイルに出力
    with open(output_path, 'w') as f:
        yaml.dump(final_profiles, f, default_flow_style=False)
        
    print(f"Developer profiles generated at: {output_path}")

# --- 実行部分 ---
if __name__ == '__main__':
    GITHUB_DATA_PATH = 'data/github_data.json'
    OUTPUT_YAML_PATH = 'configs/dev_profiles.yaml'
    generate_developer_profiles(GITHUB_DATA_PATH, OUTPUT_YAML_PATH)