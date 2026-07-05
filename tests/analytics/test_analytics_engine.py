from dataclasses import dataclass

import pytest

from basketball_vision.analytics import (
    AnalyticsConfig,
    AnalyticsEngine,
    AnalyticsFactory,
)


@dataclass(frozen=True)
class FakeBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class FakeDetection:
    bounding_box: FakeBox
    confidence: float
    class_name: str = "player"


@dataclass(frozen=True)
class FakeTrack:
    track_id: int
    detection: FakeDetection


@dataclass(frozen=True)
class FakeTrackingResult:
    frame_index: int
    tracks: tuple[FakeTrack, ...]

    def active_tracks(self) -> tuple[FakeTrack, ...]:
        return self.tracks


@dataclass(frozen=True)
class FakeAssignment:
    track_id: int
    team: str


@dataclass(frozen=True)
class FakeClassificationResult:
    assignments: tuple[FakeAssignment, ...]


@dataclass(frozen=True)
class FakeRecognition:
    track_id: int
    number: str | None


@dataclass(frozen=True)
class FakeRecognitionResult:
    recognitions: tuple[FakeRecognition, ...]


@dataclass(frozen=True)
class FakeDetectionResult:
    detections: tuple[FakeDetection, ...]


def test_analytics_engine_updates_player_and_game_statistics() -> None:
    engine = AnalyticsEngine(AnalyticsConfig(fps=10.0))

    engine.update(
        tracking_result=FakeTrackingResult(
            frame_index=0,
            tracks=(
                FakeTrack(1, FakeDetection(FakeBox(0, 0, 10, 10), 0.8)),
                FakeTrack(2, FakeDetection(FakeBox(20, 0, 30, 10), 0.6)),
            ),
        ),
        classification_result=FakeClassificationResult(
            assignments=(
                FakeAssignment(1, "blue"),
                FakeAssignment(2, "white"),
            ),
        ),
        recognition_result=FakeRecognitionResult(
            recognitions=(
                FakeRecognition(1, "23"),
                FakeRecognition(2, "11"),
            ),
        ),
        detection_result=FakeDetectionResult(
            detections=(
                FakeDetection(FakeBox(0, 0, 10, 10), 0.8, "player"),
                FakeDetection(FakeBox(5, 5, 8, 8), 0.9, "ball"),
            ),
        ),
    )

    engine.update(
        tracking_result=FakeTrackingResult(
            frame_index=1,
            tracks=(
                FakeTrack(1, FakeDetection(FakeBox(3, 4, 13, 14), 1.0)),
            ),
        ),
        classification_result=FakeClassificationResult(
            assignments=(FakeAssignment(1, "blue"),),
        ),
        recognition_result=FakeRecognitionResult(
            recognitions=(FakeRecognition(1, "23"),),
        ),
        detection_result=FakeDetectionResult(
            detections=(FakeDetection(FakeBox(3, 4, 13, 14), 1.0, "player"),),
        ),
    )

    result = engine.result()

    assert len(result.players) == 2
    assert result.game.total_frames == 2
    assert result.game.number_of_players == 2
    assert result.game.average_players_on_court == pytest.approx(1.5)
    assert result.game.ball_detections == 1
    assert result.game.team_counts == {"blue": 2, "white": 1}
    assert result.game.detection_counts == {"player": 2, "ball": 1}

    player_1 = result.players[0]
    player_2 = result.players[1]

    assert player_1.track_id == 1
    assert player_1.team == "blue"
    assert player_1.jersey_number == "23"
    assert player_1.frames_seen == 2
    assert player_1.distance_travelled_pixels == pytest.approx(5.0)

    assert player_2.track_id == 2
    assert player_2.team == "white"
    assert player_2.jersey_number == "11"
    assert player_2.frames_seen == 1


def test_analytics_engine_can_update_without_optional_results() -> None:
    engine = AnalyticsEngine(AnalyticsConfig(fps=30.0))

    engine.update(
        tracking_result=FakeTrackingResult(
            frame_index=0,
            tracks=(
                FakeTrack(1, FakeDetection(FakeBox(0, 0, 10, 10), 0.8)),
            ),
        ),
    )

    result = engine.result()

    assert len(result.players) == 1
    assert result.players[0].team is None
    assert result.players[0].jersey_number is None
    assert result.game.total_frames == 1


def test_analytics_engine_reset_clears_state() -> None:
    engine = AnalyticsEngine(AnalyticsConfig(fps=30.0))

    engine.update(
        tracking_result=FakeTrackingResult(
            frame_index=0,
            tracks=(
                FakeTrack(1, FakeDetection(FakeBox(0, 0, 10, 10), 0.8)),
            ),
        ),
    )

    engine.reset()
    result = engine.result()

    assert result.players == ()
    assert result.game.total_frames == 0


def test_factory_creates_analytics_engine() -> None:
    engine = AnalyticsFactory.create_engine(AnalyticsConfig(fps=24.0))

    assert isinstance(engine, AnalyticsEngine)
    assert engine.config.fps == 24.0
