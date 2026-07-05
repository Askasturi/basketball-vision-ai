"""CSV and JSON exporters for analytics results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from basketball_vision.analytics.exceptions import AnalyticsExportError
from basketball_vision.analytics.statistics_result import AnalyticsResult

PLAYER_STATISTICS_COLUMNS = [
    "Track ID",
    "Team",
    "Jersey Number",
    "Frames Seen",
    "Time Visible",
    "Distance",
    "Average Speed",
    "Average Confidence",
]


class AnalyticsExporter:
    """Export analytics results to CSV and JSON files."""

    def __init__(self, output_dir: Path | str = Path("outputs/analytics")) -> None:
        """Initialize exporter."""
        self.output_dir = Path(output_dir)

    def export_player_statistics(
        self,
        result: AnalyticsResult,
        filename: str = "player_statistics.csv",
    ) -> Path:
        """Export player statistics to CSV."""
        self._ensure_output_dir()
        path = self.output_dir / filename

        try:
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file, fieldnames=PLAYER_STATISTICS_COLUMNS)
                writer.writeheader()

                for player in result.players:
                    writer.writerow(
                        {
                            "Track ID": player.track_id,
                            "Team": player.team or "",
                            "Jersey Number": player.jersey_number or "",
                            "Frames Seen": player.frames_seen,
                            "Time Visible": f"{player.time_visible:.3f}",
                            "Distance": f"{player.distance_travelled_pixels:.3f}",
                            "Average Speed": f"{player.average_speed:.3f}",
                            "Average Confidence": (
                                f"{player.average_detection_confidence:.3f}"
                            ),
                        }
                    )
        except OSError as exc:
            msg = f"Failed to export player statistics to {path}."
            raise AnalyticsExportError(msg) from exc

        return path

    def export_game_statistics(
        self,
        result: AnalyticsResult,
        filename: str = "game_statistics.json",
    ) -> Path:
        """Export game statistics and player summary to JSON."""
        self._ensure_output_dir()
        path = self.output_dir / filename

        payload = self.to_json_payload(result)

        try:
            with path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2)
        except OSError as exc:
            msg = f"Failed to export game statistics to {path}."
            raise AnalyticsExportError(msg) from exc

        return path

    def export_all(self, result: AnalyticsResult) -> tuple[Path, Path]:
        """Export both CSV and JSON analytics files."""
        player_path = self.export_player_statistics(result)
        game_path = self.export_game_statistics(result)
        return player_path, game_path

    @staticmethod
    def to_json_payload(result: AnalyticsResult) -> dict[str, Any]:
        """Convert analytics result to JSON-serializable payload."""
        return {
            "frames": result.game.total_frames,
            "video_seconds": result.game.video_seconds,
            "number_of_players": result.game.number_of_players,
            "average_players_on_court": result.game.average_players_on_court,
            "ball_detections": result.game.ball_detections,
            "team_counts": result.game.team_counts,
            "estimated_ball_possession": result.game.estimated_ball_possession,
            "detection_counts": result.game.detection_counts,
            "players": [
                {
                    "track_id": player.track_id,
                    "team": player.team,
                    "jersey_number": player.jersey_number,
                    "frames_seen": player.frames_seen,
                    "first_frame": player.first_frame,
                    "last_frame": player.last_frame,
                    "time_visible": player.time_visible,
                    "average_detection_confidence": (
                        player.average_detection_confidence
                    ),
                    "average_bounding_box_area": player.average_bounding_box_area,
                    "distance_pixels": player.distance_travelled_pixels,
                    "average_speed": player.average_speed,
                    "maximum_speed": player.maximum_speed,
                }
                for player in result.players
            ],
        }

    def _ensure_output_dir(self) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            msg = f"Failed to create analytics output directory {self.output_dir}."
            raise AnalyticsExportError(msg) from exc
