from datetime import datetime


class Task:
    """
    OSSプロジェクトのタスク（IssueまたはPull Request）を表現するデータクラス。
    """

    def __init__(self, data_dict: dict):
        """
        GitHub APIから取得した辞書形式のデータからTaskオブジェクトを初期化する。

        Args:
            data_dict (dict): IssueまたはPRの単一のJSONオブジェクト。
        """
        # --- 基本情報の抽出 ---
        self.id = data_dict.get("id")
        self.number = data_dict.get("number")
        self.title = data_dict.get("title", "")
        self.body = data_dict.get("body", "")
        self.state = data_dict.get("state")  # "open" or "closed"

        # ▼▼▼【ここがエラーの修正箇所】▼▼▼
        # タイムスタンプはISO 8601形式の文字列なので、datetimeオブジェクトに変換します。
        created_at_str = data_dict.get("created_at")
        updated_at_str = data_dict.get("updated_at")

        # タイムスタンプ文字列をパースして、self.created_at等の属性として設定します。
        self.created_at = self.parse_datetime(created_at_str)
        self.updated_at = self.parse_datetime(updated_at_str)
        # ▲▲▲【ここまでがエラーの修正箇所】▲▲▲

        # --- ラベル情報の抽出 ---
        # ラベルは {'name': 'bug', ...} という辞書のリストになっています。
        self.labels = [
            label["name"] for label in data_dict.get("labels", []) if "name" in label
        ]

        # --- ユーザー情報の抽出 ---
        self.author = data_dict.get("user", {}).get("login")

        # --- その他のメタ情報 ---
        self.comments = data_dict.get("comments", 0)
        self.html_url = data_dict.get("html_url")

        # --- シミュレーション中に更新される状態 ---
        self.status = "todo"  # 'todo', 'in_progress', 'done'
        self.assigned_to = None
        # get_github_data.pyがPRのファイルリストを取得している場合、ここで設定できます。
        self.changed_files = [
            f.get("filename") for f in data_dict.get("files", []) if f.get("filename")
        ]

    @staticmethod
    def parse_datetime(timestamp_str: str | None) -> datetime | None:
        """
        ISO 8601形式のタイムスタンプ文字列をdatetimeオブジェクトに変換します。
        末尾の'Z'はUTCを示しますが、Pythonのバージョンによって扱いが異なるため、
        安全に処理できるようにしています。
        """
        if not timestamp_str:
            return None
        try:
            # Python 3.11+ は 'Z' を直接扱えます。
            # 古いバージョン(3.7-3.10)のために、'Z'をUTCオフセット'+00:00'に置換します。
            if timestamp_str.endswith("Z"):
                timestamp_str = timestamp_str[:-1] + "+00:00"
            return datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            # パースに失敗した場合はNoneを返します。
            print(f"Warning: Could not parse timestamp '{timestamp_str}'.")
            return None

    @classmethod
    def from_dict(cls, data_dict: dict):
        """辞書からTaskインスタンスを生成するためのファクトリメソッド。"""
        return cls(data_dict)

    def __repr__(self):
        return f"<Task #{self.number}: {self.title}>"
