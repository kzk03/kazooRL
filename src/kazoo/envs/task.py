class Task:
    def __init__(self, id, title, author, complexity, created_at, labels):
        self.id = id
        self.title = title
        self.author = author
        self.complexity = complexity
        self.created_at = created_at
        self.labels = labels
        self.state = None
