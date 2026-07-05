"""Shared classification types."""

from enum import StrEnum


class TeamLabel(StrEnum):
    """Supported team labels."""

    TEAM_A = "team_a"
    TEAM_B = "team_b"
    UNKNOWN = "unknown"


class ClassifierType(StrEnum):
    """Supported classifier implementations."""

    COLOR = "color"
