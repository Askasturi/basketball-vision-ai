from enum import StrEnum


class TrackState(StrEnum):
    NEW = "new"
    ACTIVE = "active"
    LOST = "lost"
    REMOVED = "removed"


class TrackerType(StrEnum):
    SIMPLE = "simple"
