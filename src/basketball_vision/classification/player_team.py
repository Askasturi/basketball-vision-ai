"""Data models for player team classification."""

from dataclasses import dataclass
from typing import Any

from basketball_vision.classification.types import TeamLabel


@dataclass(frozen=True, slots=True)
class PlayerTeamAssignment:
    """Team assignment for one tracked player."""

    track_id: int
    team: TeamLabel
    confidence: float
    track: Any


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Classification output for one video frame."""

    frame_index: int
    timestamp: float | None
    assignments: tuple[PlayerTeamAssignment, ...]

    def __len__(self) -> int:
        """Return the number of assignments."""
        return len(self.assignments)

    def __iter__(self):
        """Iterate over assignments."""
        return iter(self.assignments)

    def get_assignment(self, track_id: int) -> PlayerTeamAssignment | None:
        """Return an assignment by track id."""
        for assignment in self.assignments:
            if assignment.track_id == track_id:
                return assignment
        return None
