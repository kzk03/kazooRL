from enum import Enum, auto


class Action(Enum):
    # Issue-related
    ASSIGN_TASK = auto()
    CLOSE_ISSUE = auto()

    # PR-related
    SUBMIT_PULL_REQUEST = auto()
    REQUEST_REVIEW = auto()
    APPROVE_PULL_REQUEST = auto()  # レビューでのApprove
    MERGE_PULL_REQUEST = auto()  # マージ行為

    # General
    DO_NOTHING = auto()
