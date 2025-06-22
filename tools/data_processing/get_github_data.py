import os
import requests
import gzip
import json
import time
from datetime import datetime, timedelta

# --- 設定 ---
# 1. 分析対象のリポジトリ
REPO_NAME = "docker/compose"

# 2. 取得する期間
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2023, 12, 31)

# 3. 保存するファイルのベース名
OUTPUT_FILE_BASENAME = "gharchive_docker_compose_events"
# --- ---

def download_and_filter_archive(output_basename):
    current_date = START_DATE
    
    output_file_handle = None
    current_file_year = None

    try:
        while current_date <= END_DATE:
            if current_date.year != current_file_year:
                if output_file_handle:
                    output_file_handle.close()
                    print(f"年が変わったため、{current_file_year}年のファイルを保存しました。")
                
                current_file_year = current_date.year
                output_filename = f"{output_basename}_{current_file_year}.jsonl"
                print(f"\n>>>> {current_file_year}年のデータを処理します。保存先: {output_filename} <<<<\n")
                
                output_file_handle = open(output_filename, 'w', encoding='utf-8')

            for hour in range(24):
                url = f"https://data.gharchive.org/{current_date.strftime('%Y-%m-%d')}-{hour}.json.gz"
                
                print(f"ダウンロード中: {url}")
                
                for attempt in range(3):
                    try:
                        response = requests.get(url, stream=True)
                        response.raise_for_status()
                        
                        with gzip.GzipFile(fileobj=response.raw) as gz:
                            for line in gz:
                                event = json.loads(line)
                                if event.get('repo', {}).get('name') == REPO_NAME:
                                    output_file_handle.write(json.dumps(event) + '\n')
                        break
                                
                    except requests.exceptions.RequestException as e:
                        print(f"ダウンロードエラー (試行 {attempt + 1}/3): {e}")
                        if attempt < 2:
                            time.sleep(5)
                        else:
                            print(f"3回失敗したため、ファイル {url} をスキップします。")
                
                time.sleep(1)

            current_date += timedelta(days=1)
            
    finally:
        if output_file_handle:
            output_file_handle.close()
            print(f"\n最後のファイル ({current_file_year}年) を保存しました。")
        
    print(f"全ての処理が完了しました。")

if __name__ == "__main__":
    download_and_filter_archive(OUTPUT_FILE_BASENAME)