"""Game-level statistics calculation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from basketball_vision.analytics.statistics_result import GameStatistics


@dataclass(slots=True)
class GameStatisticsAccumulator:
    """Mutable accumulator for game-level statistics."""

    fps: float
    total_frames: int = 0
    ball_detections: int = 0
    team_counts: Counter[str] = field(default_factory=Counter)
    detection_counts: Counter[str] = field(default_factory=Counter)
    players_per_frame: list[int] = field(default_factory=list)

    def update(
        self,
        *,
        players_on_court: int,
        teams: list[str],
        detection_labels: list[str] | None = None,
        ball_detections: int = 0,
    ) -> None:
        """Update game statistics from one frame."""
        self.total_frames += 1
        self.players_per_frame.append(players_on_court)
        self.ball_detections += ball_detections

        for team in teams:
            self.team_counts[str(team)] += 1

        for label in detection_labels or []:
            self.detection_counts[str(label)] += 1

    def to_result(self, number_of_players: int) -> GameStatistics:
        """Convert accumulated statistics to an immutable result."""
        average_players = (
            sum(self.players_per_frame) / len(self.players_per_frame)
            if self.players_per_frame
            else 0.0
        )

        total_team_observations = sum(self.team_counts.values())
        possession = {
            team: count / total_team_observations
            for team, count in self.team_counts.items()
        }

        return GameStatistics(
            total_frames=self.total_frames,
            video_seconds=self.total_frames / self.fps,
            number_of_players=number_of_players,
            average_players_on_court=average_players,
            ball_detections=self.ball_detections,
            team_counts=dict(self.team_counts),
            estimated_ball_possession=possession,
            detection_counts=dict(self.detection_counts),
        )
