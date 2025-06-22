import gzip
import json
import os
import time
from datetime import datetime, timedelta

import requests

# --- 設定 ---
REPO_NAME = "docker/compose"
START_DATE = datetime(2020, 8, 21)
END_DATE = datetime(2025, 6, 19)
OUTPUT_FILE_BASENAME = "gharchive_docker_compose_events"
OUTPUT_DIR = "./results"

RETRY_WAIT_SECONDS = 5  # 最初の再試行待ち時間
MAX_RETRY_WAIT_SECONDS = 300 # 最大再試行待ち時間 (5分)
MAX_RETRIES = 5 # 最大リトライ回数

# GitHub APIのレートリミット関連ヘッダー (GitHub Archiveには直接関係ないが、API利用時の参考として)
# X-RateLimit-Limit: リクエスト上限
# X-RateLimit-Remaining: 残りリクエスト数
# X-RateLimit-Reset: リセットされるUNIXタイムスタンプ
# --- ---

FAILED_LOG_PATH = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE_BASENAME}_failed_downloads.log")

def log_failed_download(date, hour, url, reason):
    """ダウンロード失敗したURLと理由をログに記録する"""
    with open(FAILED_LOG_PATH, 'a', encoding='utf-8') as f:
        timestamp = f"{date.strftime('%Y-%m-%d')}-{hour:02d}"
        f.write(f"{timestamp}\t{url}\t{reason}\n")

def download_and_filter_archive(output_basename):
    """GitHub Archiveからデータをダウンロードし、指定リポジトリのイベントをフィルタリングして保存する"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    current_date = START_DATE
    output_file_handle = None
    current_file_month = None

    try:
        while current_date <= END_DATE:
            year_month = current_date.strftime("%Y-%m")

            # 月が変わったら新しい出力ファイルを開く
            if year_month != current_file_month:
                if output_file_handle:
                    output_file_handle.close()
                    print(f"\n✅ {current_file_month} のファイルを保存しました。")
                
                current_file_month = year_month
                output_filename = os.path.join(OUTPUT_DIR, f"{output_basename}_{current_file_month}.jsonl")
                print(f"\n>>>> {current_file_month} のデータを処理します。保存先: {output_filename} <<<<\n")
                output_file_handle = open(output_filename, 'a', encoding='utf-8') # 'a' で追記モードに変更

            for hour in range(24):
                url = f"https://data.gharchive.org/{current_date.strftime('%Y-%m-%d')}-{hour}.json.gz"
                print(f"🔗 {current_date.strftime('%Y-%m-%d')} {hour:02d}時 のデータ取得開始 → {url}")
                
                attempt = 0
                current_wait_time = RETRY_WAIT_SECONDS
                download_successful = False # ダウンロード成功フラグ

                while attempt < MAX_RETRIES:
                    try:
                        response = requests.get(url, stream=True, timeout=30) # タイムアウトを追加
                        response.raise_for_status() # 2xx 以外の場合、HTTPError を発生させる
                        
                        with gzip.GzipFile(fileobj=response.raw) as gz:
                            for line in gz:
                                try:
                                    event = json.loads(line)
                                    if event.get('repo', {}).get('name') == REPO_NAME:
                                        output_file_handle.write(json.dumps(event) + '\n')
                                except json.JSONDecodeError:
                                    # JSONデコードエラーはスキップし、ログに記録しない（データが壊れている可能性）
                                    continue
                        
                        print(f"✅ 成功: {url}")
                        download_successful = True
                        break # 成功したのでループを抜ける
                    
                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 404:
                            # 404 Not Found の場合
                            print(f"⚠️ URLが見つかりません (404)。スキップします: {url}")
                            log_failed_download(current_date, hour, url, "Not Found (404)")
                            download_successful = True # 見つからなかったが、エラーとしてリトライはしない
                            break # スキップして次の時間へ
                        elif e.response.status_code in [403, 429]:
                            # 403 Forbidden (API制限など)、429 Too Many Requests (レートリミット) の場合
                            attempt += 1
                            print(f"⚠️ API制限または一時的なサーバーエラー (ステータスコード: {e.response.status_code})。試行 {attempt}/{MAX_RETRIES}: {e}")
                            
                            # GitHub APIのレートリミットヘッダーがあれば利用
                            retry_after = e.response.headers.get("Retry-After")
                            if retry_after:
                                try:
                                    wait_time = int(retry_after) + 5 # Retry-Afterに少し余裕を持たせる
                                    current_wait_time = min(wait_time, MAX_RETRY_WAIT_SECONDS)
                                    print(f"「Retry-After」ヘッダーに基づいて {current_wait_time}秒待機します...")
                                except ValueError:
                                    print(f"「Retry-After」ヘッダーの値が無効です。デフォルトで {current_wait_time}秒待機します...")
                                time.sleep(current_wait_time)
                            else:
                                print(f"{current_wait_time}秒待機して再試行します...")
                                time.sleep(current_wait_time)
                                current_wait_time = min(current_wait_time * 2, MAX_RETRY_WAIT_SECONDS) # Exponential backoff
                        else:
                            # その他のHTTPエラー（5xx系など）
                            attempt += 1
                            print(f"⚠️ その他のHTTPエラー (ステータスコード: {e.response.status_code})。試行 {attempt}/{MAX_RETRIES}: {e}")
                            print(f"{current_wait_time}秒待機して再試行します...")
                            time.sleep(current_wait_time)
                            current_wait_time = min(current_wait_time * 2, MAX_RETRY_WAIT_SECONDS)

                    except requests.exceptions.Timeout:
                        attempt += 1
                        print(f"⚠️ タイムアウトエラー (試行 {attempt}/{MAX_RETRIES}): {url}")
                        print(f"{current_wait_time}秒待機して再試行します...")
                        time.sleep(current_wait_time)
                        current_wait_time = min(current_wait_time * 2, MAX_RETRY_WAIT_SECONDS)
                    
                    except requests.exceptions.ConnectionError as e:
                        attempt += 1
                        print(f"⚠️ 接続エラー (試行 {attempt}/{MAX_RETRIES}): {e}")
                        print(f"{current_wait_time}秒待機して再試行します...")
                        time.sleep(current_wait_time)
                        current_wait_time = min(current_wait_time * 2, MAX_RETRY_WAIT_SECONDS)

                    except requests.exceptions.RequestException as e:
                        # 上記以外のrequests関連の一般エラー
                        attempt += 1
                        print(f"⚠️ その他のリクエストエラー (試行 {attempt}/{MAX_RETRIES}): {e}")
                        print(f"{current_wait_time}秒待機して再試行します...")
                        time.sleep(current_wait_time)
                        current_wait_time = min(current_wait_time * 2, MAX_RETRY_WAIT_SECONDS)
                
                # リトライ回数をオーバーしても成功しなかった場合
                if not download_successful:
                    print(f"❌ 最大リトライ回数に達しました。ダウンロードをスキップします: {url}")
                    log_failed_download(current_date, hour, url, "Max retries exceeded or unhandled error")

                time.sleep(3) # 次の時間へのインターバル。サーバー負荷軽減のため

            current_date += timedelta(days=1)
            
    finally:
        # スクリプト終了時に開いているファイルを必ずクローズする
        if output_file_handle:
            output_file_handle.close()
            print(f"\n✅ 最後のファイル ({current_file_month}) を保存しました。")
        
    print("🎉 全ての処理が完了しました。")

if __name__ == "__main__":
    download_and_filter_archive(OUTPUT_FILE_BASENAME)