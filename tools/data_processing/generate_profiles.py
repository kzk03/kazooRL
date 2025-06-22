import glob
import json
import os
from collections import defaultdict

import yaml


def generate_developer_profiles(data_dir, output_path):
    """
    指定されたディレクトリにある全ての .jsonl ファイルを読み込み、
    開発者ごとの静的な特徴量を事前計算して、dev_profiles.yamlとして出力する。

    計算する特徴量:
    - label_affinity: どのラベルのタスクをどれくらいの割合で完了させたか
    - touched_files: これまでに編集したことのあるファイルの一覧
    - total_merged_prs: マージされたPRの総数
    """
    print(f"Starting to generate developer profiles from directory: {data_dir}")

    # data_dir内の全ての .jsonl ファイルを取得
    # 例: "results/gharchive_docker_compose_events_2020-08.jsonl"
    jsonl_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    if not jsonl_files:
        print(f"Error: No .jsonl files found in directory: {data_dir}")
        return

    print(f"Found {len(jsonl_files)} files to process.")

    dev_stats = defaultdict(lambda: {
        'label_counts': defaultdict(int),
        'touched_files': set(),
        'merged_pr_count': 0
    })

    for file_path in sorted(jsonl_files):
        print(f"Processing file: {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        # 壊れた行はスキップ
                        continue
                    
                    # # 【検証のための変更後】
                    # if event.get('type') == 'PullRequestEvent' and \
                    # event.get('payload', {}).get('action') == 'closed':
                    #     # PRがクローズされたイベントを処理
                    #     continue

                    if event.get('type') == 'PullRequestEvent' and \
                       event.get('payload', {}).get('action') == 'closed' and \
                       event.get('payload', {}).get('pull_request', {}).get('merged'):
                        
                        pr = event['payload']['pull_request']
                        developer = pr.get('user', {}).get('login')
                        if not developer:
                            continue

                        dev_stats[developer]['merged_pr_count'] += 1

                        for label in pr.get('labels', []):
                            if 'name' in label:
                                dev_stats[developer]['label_counts'][label['name']] += 1
                        
                        # ファイルリストの取得 (get_github_data.pyでの取得が前提)
                        for file_info in pr.get('files', []):
                             if 'filename' in file_info:
                                dev_stats[developer]['touched_files'].add(file_info['filename'])
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")


    final_profiles = {}
    for dev, stats in dev_stats.items():
        total_labels = sum(stats['label_counts'].values())
        label_affinity = {
            label: count / total_labels for label, count in stats['label_counts'].items()
        } if total_labels > 0 else {}
        
        final_profiles[dev] = {
            'skills': ['python'], 
            'label_affinity': label_affinity,
            'touched_files': sorted(list(stats['touched_files'])),
            'total_merged_prs': stats['merged_pr_count']
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(final_profiles, f, default_flow_style=False, allow_unicode=True)
        
    print(f"✅ Successfully generated developer profiles for {len(final_profiles)} developers at: {output_path}")

if __name__ == '__main__':
    # データ取得スクリプトが出力したディレクトリを指定
    INPUT_DATA_DIR = './data/2019' 
    OUTPUT_YAML_PATH = 'configs/dev_profiles.yaml'
    generate_developer_profiles(INPUT_DATA_DIR, OUTPUT_YAML_PATH)
