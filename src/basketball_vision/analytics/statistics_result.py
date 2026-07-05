"""Immutable analytics result models."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlayerStatistics:
    """Statistics for one tracked player."""

    track_id: int
    team: str | None
    jersey_number: str | None
    frames_seen: int
    first_frame: int
    last_frame: int
    time_visible: float
    average_detection_confidence: float
    average_bounding_box_area: float
    distance_travelled_pixels: float
    average_speed: float
    maximum_speed: float


@dataclass(frozen=True, slots=True)
class GameStatistics:
    """Aggregate game statistics."""

    total_frames: int
    video_seconds: float
    number_of_players: int
    average_players_on_court: float
    ball_detections: int
    team_counts: dict[str, int]
    estimated_ball_possession: dict[str, float]
    detection_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class AnalyticsResult:
    """Full analytics result."""

    players: tuple[PlayerStatistics, ...]
    game: GameStatistics
