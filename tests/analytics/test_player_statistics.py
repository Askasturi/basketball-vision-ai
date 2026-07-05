from dataclasses import dataclass

import pytest

from basketball_vision.analytics import PlayerStatisticsAccumulator


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


@dataclass(frozen=True)
class FakeTrack:
    track_id: int
    detection: FakeDetection


def test_player_statistics_accumulator_tracks_player_metrics() -> None:
    accumulator = PlayerStatisticsAccumulator(track_id=7, fps=10.0)

    accumulator.update(
        frame_index=0,
        track=FakeTrack(
            track_id=7,
            detection=FakeDetection(FakeBox(0, 0, 10, 10), 0.8),
        ),
        team="blue",
        jersey_number="23",
    )
    accumulator.update(
        frame_index=1,
        track=FakeTrack(
            track_id=7,
            detection=FakeDetection(FakeBox(3, 4, 13, 14), 1.0),
        ),
        team="blue",
        jersey_number="23",
    )

    result = accumulator.to_result()

    assert result.track_id == 7
    assert result.team == "blue"
    assert result.jersey_number == "23"
    assert result.frames_seen == 2
    assert result.first_frame == 0
    assert result.last_frame == 1
    assert result.time_visible == pytest.approx(0.2)
    assert result.average_detection_confidence == pytest.approx(0.9)
    assert result.average_bounding_box_area == pytest.approx(100.0)
    assert result.distance_travelled_pixels == pytest.approx(5.0)
    assert result.average_speed == pytest.approx(25.0)
    assert result.maximum_speed == pytest.approx(50.0)


def test_player_statistics_supports_tuple_box() -> None:
    accumulator = PlayerStatisticsAccumulator(track_id=1, fps=30.0)

    accumulator.update(
        frame_index=0,
        track=FakeDetection((0, 0, 20, 10), 0.5),
    )

    result = accumulator.to_result()

    assert result.average_bounding_box_area == pytest.approx(200.0)
