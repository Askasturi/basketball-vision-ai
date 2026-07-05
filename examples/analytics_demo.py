"""Run a deterministic analytics demo and export video/statistics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from basketball_vision.analytics import (
    AnalyticsConfig,
    AnalyticsEngine,
    AnalyticsExporter,
)
from basketball_vision.visualization.renderer import Renderer


@dataclass(frozen=True)
class DemoBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class DemoDetection:
    bounding_box: DemoBox
    confidence: float
    class_name: str = "player"


@dataclass(frozen=True)
class DemoTrack:
    track_id: int
    detection: DemoDetection
    team: str
    jersey_number: str

    @property
    def bounding_box(self) -> DemoBox:
        return self.detection.bounding_box

    @property
    def confidence(self) -> float:
        return self.detection.confidence

    @property
    def number(self) -> str:
        return self.jersey_number


@dataclass(frozen=True)
class DemoTrackingResult:
    frame_index: int
    timestamp: float
    tracks: tuple[DemoTrack, ...]

    def active_tracks(self) -> tuple[DemoTrack, ...]:
        return self.tracks


@dataclass(frozen=True)
class DemoAssignment:
    track_id: int
    team: str


@dataclass(frozen=True)
class DemoClassificationResult:
    assignments: tuple[DemoAssignment, ...]


@dataclass(frozen=True)
class DemoRecognition:
    track_id: int
    number: str


@dataclass(frozen=True)
class DemoRecognitionResult:
    recognitions: tuple[DemoRecognition, ...]


@dataclass(frozen=True)
class DemoDetectionResult:
    detections: tuple[DemoDetection, ...]


@dataclass(frozen=True)
class DemoRenderResult:
    frame_index: int
    objects: tuple[DemoTrack, ...]
    analytics: object


def build_frame(width: int, height: int) -> np.ndarray:
    """Create a simple blank basketball-court-style frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (40, 120, 40)
    cv2.rectangle(frame, (20, 20), (width - 20,
                  height - 20), (220, 220, 220), 2)
    cv2.line(frame, (width // 2, 20),
             (width // 2, height - 20), (220, 220, 220), 2)
    cv2.circle(frame, (width // 2, height // 2), 45, (220, 220, 220), 2)
    return frame


def build_tracks(frame_index: int) -> tuple[DemoTrack, ...]:
    """Create deterministic moving player tracks."""
    return (
        DemoTrack(
            track_id=1,
            detection=DemoDetection(
                bounding_box=DemoBox(
                    x1=50 + frame_index * 2,
                    y1=60,
                    x2=90 + frame_index * 2,
                    y2=140,
                ),
                confidence=0.92,
            ),
            team="blue",
            jersey_number="23",
        ),
        DemoTrack(
            track_id=2,
            detection=DemoDetection(
                bounding_box=DemoBox(
                    x1=260 - frame_index,
                    y1=80,
                    x2=300 - frame_index,
                    y2=160,
                ),
                confidence=0.88,
            ),
            team="white",
            jersey_number="11",
        ),
    )


def run_demo(
    output_video: Path = Path("outputs/demo/analytics_demo.mp4"),
    analytics_dir: Path = Path("outputs/analytics"),
    frames: int = 30,
    fps: float = 30.0,
) -> tuple[Path, Path, Path]:
    """Run analytics demo and return created output paths."""
    output_video.parent.mkdir(parents=True, exist_ok=True)
    analytics_dir.mkdir(parents=True, exist_ok=True)

    width = 420
    height = 240

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        msg = f"Could not open video writer for {output_video}"
        raise RuntimeError(msg)

    engine = AnalyticsEngine(AnalyticsConfig(
        fps=fps, output_dir=analytics_dir))
    renderer = Renderer()

    for frame_index in range(frames):
        frame = build_frame(width=width, height=height)
        tracks = build_tracks(frame_index)

        tracking = DemoTrackingResult(
            frame_index=frame_index,
            timestamp=frame_index / fps,
            tracks=tracks,
        )
        classification = DemoClassificationResult(
            assignments=tuple(
                DemoAssignment(track_id=track.track_id, team=track.team)
                for track in tracks
            )
        )
        recognition = DemoRecognitionResult(
            recognitions=tuple(
                DemoRecognition(
                    track_id=track.track_id,
                    number=track.jersey_number,
                )
                for track in tracks
            )
        )
        detection = DemoDetectionResult(
            detections=tuple(track.detection for track in tracks)
        )

        engine.update(
            tracking_result=tracking,
            classification_result=classification,
            recognition_result=recognition,
            detection_result=detection,
        )

        render_result = DemoRenderResult(
            frame_index=frame_index,
            objects=tracks,
            analytics=engine.result(),
        )
        rendered = renderer.render(
            frame=frame, pipeline_frame_result=render_result)
        writer.write(rendered)

    writer.release()

    exporter = AnalyticsExporter(output_dir=analytics_dir)
    player_csv, game_json = exporter.export_all(engine.result())

    return output_video, player_csv, game_json


def main() -> None:
    """Run the analytics demo."""
    video_path, player_csv, game_json = run_demo()
    print(f"Wrote video: {video_path}")
    print(f"Wrote player statistics: {player_csv}")
    print(f"Wrote game statistics: {game_json}")


if __name__ == "__main__":
    main()
