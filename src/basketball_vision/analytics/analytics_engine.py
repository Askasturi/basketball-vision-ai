"""Analytics engine for frame-by-frame basketball statistics."""

from __future__ import annotations

from typing import Any

from basketball_vision.analytics.config import AnalyticsConfig
from basketball_vision.analytics.game_statistics import GameStatisticsAccumulator
from basketball_vision.analytics.player_statistics import PlayerStatisticsAccumulator
from basketball_vision.analytics.statistics_result import AnalyticsResult


class AnalyticsEngine:
    """Compute basketball analytics from existing pipeline results."""

    def __init__(self, config: AnalyticsConfig | None = None) -> None:
        """Initialize the analytics engine."""
        self.config = config or AnalyticsConfig()
        self._players: dict[int, PlayerStatisticsAccumulator] = {}
        self._game = GameStatisticsAccumulator(fps=self.config.fps)

    def update(
        self,
        *,
        tracking_result: Any,
        classification_result: Any | None = None,
        recognition_result: Any | None = None,
        detection_result: Any | None = None,
    ) -> None:
        """Update analytics from one processed frame."""
        active_tracks = tuple(tracking_result.active_tracks())

        classification_by_track = self._classification_by_track(
            classification_result,
        )
        recognition_by_track = self._recognition_by_track(recognition_result)

        teams: list[str] = []

        for track in active_tracks:
            track_id = int(track.track_id)
            assignment = classification_by_track.get(track_id)
            recognition = recognition_by_track.get(track_id)

            team = self._extract_team(assignment)
            jersey_number = self._extract_number(recognition)

            if team is not None:
                teams.append(team)

            if track_id not in self._players:
                self._players[track_id] = PlayerStatisticsAccumulator(
                    track_id=track_id,
                    fps=self.config.fps,
                )

            self._players[track_id].update(
                frame_index=int(tracking_result.frame_index),
                track=track,
                team=team,
                jersey_number=jersey_number,
            )

        detection_labels = self._extract_detection_labels(detection_result)
        ball_detections = sum(
            1 for label in detection_labels if label.lower() == "ball"
        )

        self._game.update(
            players_on_court=len(active_tracks),
            teams=teams,
            detection_labels=detection_labels,
            ball_detections=ball_detections,
        )

    def result(self) -> AnalyticsResult:
        """Return immutable analytics results."""
        players = tuple(
            accumulator.to_result()
            for _, accumulator in sorted(self._players.items())
        )
        return AnalyticsResult(
            players=players,
            game=self._game.to_result(number_of_players=len(players)),
        )

    def reset(self) -> None:
        """Reset all accumulated analytics."""
        self._players.clear()
        self._game = GameStatisticsAccumulator(fps=self.config.fps)

    @staticmethod
    def _classification_by_track(
        classification_result: Any | None,
    ) -> dict[int, Any]:
        if classification_result is None:
            return {}

        assignments = getattr(classification_result, "assignments", ())
        return {
            int(assignment.track_id): assignment
            for assignment in assignments
        }

    @staticmethod
    def _recognition_by_track(
        recognition_result: Any | None,
    ) -> dict[int, Any]:
        if recognition_result is None:
            return {}

        recognitions = getattr(recognition_result, "recognitions", ())
        return {
            int(recognition.track_id): recognition
            for recognition in recognitions
        }

    @staticmethod
    def _extract_team(assignment: Any | None) -> str | None:
        if assignment is None:
            return None

        team = getattr(assignment, "team", None)
        if team is None:
            return None

        value = getattr(team, "value", team)
        return str(value)

    @staticmethod
    def _extract_number(recognition: Any | None) -> str | None:
        if recognition is None:
            return None

        number = getattr(recognition, "number", None)
        if number is None:
            return None

        return str(number)

    @staticmethod
    def _extract_detection_labels(detection_result: Any | None) -> list[str]:
        if detection_result is None:
            return []

        detections = getattr(detection_result, "detections", None)
        if detections is None:
            detections = getattr(detection_result, "objects", None)
        if detections is None:
            detections = detection_result if isinstance(
                detection_result, list) else ()

        labels: list[str] = []
        for detection in detections:
            label = getattr(detection, "class_name", None)
            if label is None:
                label = getattr(detection, "label", None)
            if label is None:
                label = getattr(detection, "name", None)
            if label is not None:
                labels.append(str(label))

        return labels
