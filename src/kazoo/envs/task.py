import random  # 乱数ライブラリ


class Task:
    def __init__(
        self, id, name, required_skills, complexity, deadline, dependencies=None
    ):
        # --- 初期化メソッド ---
        self.id = id  # タスクの一意なID
        self.name = name  # タスクの名前（例: "Fix login bug"）
        self.required_skills = (
            required_skills  # タスクを完了するために必要なスキル（リスト）
        )
        self.complexity = complexity  # タスクの複雑さ（完了までにかかる時間などを決定）
        self.deadline = deadline  # タスクの完了期限
        self.dependencies = (
            dependencies or []
        )  # このタスクが依存する他のタスクのIDリスト

        # --- タスクの状態管理用属性 ---
        self.status = "todo"  # タスクの現在の状態（todo, in_progress, done）
        self.assigned_to = None  # 担当している開発者のID
        self.start_time = None  # タスクが開始された時間
        self.completion_time = None  # タスクが完了した時間

    @classmethod
    def from_dict(cls, data):
        # --- 辞書データからTaskインスタンスを生成するクラスメソッド ---
        # backlog.jsonのような外部データから簡単にTaskオブジェクトを作れるようにする
        return cls(
            id=data["id"],
            name=data["name"],
            required_skills=data["required_skills"],
            complexity=data["complexity"],
            deadline=data.get(
                "deadline", 100
            ),  # デッドラインがなければデフォルト値100を設定
        )

    def __repr__(self):
        # --- print()などで表示される際の文字列表現を定義 ---
        return f"Task(id={self.id}, name='{self.name}', status='{self.status}')"
