
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pytest

from basketball_vision.recognition import (
    NumberRecognitionStatus,
    RecognitionConfig,
    RecognitionInputError,
    SimpleNumberRecognizer,
)


class DummyTrackStatus(StrEnum):
    ACTIVE = "active"
    LOST = "lost"
    REMOVED = "removed"


@dataclass(frozen=True)
class DummyBoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class DummyTrack:
    track_id: int
    bounding_box: DummyBoundingBox
    status: DummyTrackStatus = DummyTrackStatus.ACTIVE


@dataclass(frozen=True)
class DummyTrackingResult:
    frame_index: int
    timestamp: float
    tracks: tuple[DummyTrack, ...]


@dataclass(frozen=True)
class DummyTeamAssignment:
    track_id: int
    team_id: int


@dataclass(frozen=True)
class DummyClassificationResult:
    frame_index: int
    timestamp: float
    assignments: tuple[DummyTeamAssignment, ...]


def make_frame():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_recognizer_returns_unknown_without_mapping():
    tracking_result = DummyTrackingResult(
        frame_index=10,
        timestamp=2.5,
        tracks=(DummyTrack(track_id=1, bounding_box=DummyBoundingBox(10, 10, 40, 50)),),
    )
    recognizer = SimpleNumberRecognizer()

    result = recognizer.recognize(make_frame(), tracking_result)

    assert result.frame_index == 10
    assert result.timestamp == 2.5
    assert len(result.recognitions) == 1
    assert result.recognitions[0].track_id == 1
    assert result.recognitions[0].number is None
    assert result.recognitions[0].confidence == 0.0
    assert result.recognitions[0].status is NumberRecognitionStatus.UNKNOWN


def test_recognizer_uses_configured_mapping():
    tracking_result = DummyTrackingResult(
        frame_index=10,
        timestamp=2.5,
        tracks=(
            DummyTrack(track_id=1, bounding_box=DummyBoundingBox(10, 10, 40, 50)),
            DummyTrack(track_id=2, bounding_box=DummyBoundingBox(50, 10, 80, 50)),
        ),
    )
    recognizer = SimpleNumberRecognizer(
        config=RecognitionConfig(track_id_to_number={1: "23", 2: "7"})
    )

    result = recognizer.recognize(make_frame(), tracking_result)

    assert [recognition.number for recognition in result.recognitions] == ["23", "7"]
    assert all(
        recognition.status is NumberRecognitionStatus.RECOGNIZED
        for recognition in result.recognitions
    )
    assert all(recognition.confidence == 1.0 for recognition in result.recognitions)


def test_recognizer_attaches_team_assignment():
    tracking_result = DummyTrackingResult(
        frame_index=10,
        timestamp=2.5,
        tracks=(DummyTrack(track_id=1, bounding_box=DummyBoundingBox(10, 10, 40, 50)),),
    )
    classification_result = DummyClassificationResult(
        frame_index=10,
        timestamp=2.5,
        assignments=(DummyTeamAssignment(track_id=1, team_id=2),),
    )

    recognizer = SimpleNumberRecognizer(
        config=RecognitionConfig(track_id_to_number={1: "23"})
    )
    result = recognizer.recognize(make_frame(), tracking_result, classification_result)

    assert result.recognitions[0].team_assignment == DummyTeamAssignment(
        track_id=1,
        team_id=2,
    )


def test_recognizer_skips_lost_and_removed_tracks_by_default():
    tracking_result = DummyTrackingResult(
        frame_index=0,
        timestamp=0.0,
        tracks=(
            DummyTrack(
                track_id=1,
                bounding_box=DummyBoundingBox(10, 10, 40, 50),
                status=DummyTrackStatus.ACTIVE,
            ),
            DummyTrack(
                track_id=2,
                bounding_box=DummyBoundingBox(10, 10, 40, 50),
                status=DummyTrackStatus.LOST,
            ),
            DummyTrack(
                track_id=3,
                bounding_box=DummyBoundingBox(10, 10, 40, 50),
                status=DummyTrackStatus.REMOVED,
            ),
        ),
    )

    recognizer = SimpleNumberRecognizer(
        config=RecognitionConfig(track_id_to_number={1: "23", 2: "7", 3: "11"})
    )
    result = recognizer.recognize(make_frame(), tracking_result)

    assert [recognition.track_id for recognition in result.recognitions] == [1]


def test_recognizer_can_include_lost_and_removed_tracks():
    tracking_result = DummyTrackingResult(
        frame_index=0,
        timestamp=0.0,
        tracks=(
            DummyTrack(
                track_id=1,
                bounding_box=DummyBoundingBox(10, 10, 40, 50),
                status=DummyTrackStatus.LOST,
            ),
            DummyTrack(
                track_id=2,
                bounding_box=DummyBoundingBox(10, 10, 40, 50),
                status=DummyTrackStatus.REMOVED,
            ),
        ),
    )

    recognizer = SimpleNumberRecognizer(
        config=RecognitionConfig(
            include_lost_tracks=True,
            include_removed_tracks=True,
            track_id_to_number={1: "23", 2: "7"},
        )
    )
    result = recognizer.recognize(make_frame(), tracking_result)

    assert [recognition.track_id for recognition in result.recognitions] == [1, 2]


def test_invalid_frame_type_raises():
    recognizer = SimpleNumberRecognizer()
    tracking_result = DummyTrackingResult(frame_index=0, timestamp=0.0, tracks=())

    with pytest.raises(RecognitionInputError):
        recognizer.recognize("not-frame", tracking_result)


def test_empty_frame_raises():
    recognizer = SimpleNumberRecognizer()
    tracking_result = DummyTrackingResult(frame_index=0, timestamp=0.0, tracks=())

    with pytest.raises(RecognitionInputError):
        recognizer.recognize(np.array([]), tracking_result)


def test_missing_tracks_raises():
    recognizer = SimpleNumberRecognizer()

    with pytest.raises(RecognitionInputError):
        recognizer.recognize(make_frame(), object())


def test_missing_bounding_box_raises():
    @dataclass(frozen=True)
    class BadTrack:
        track_id: int

    @dataclass(frozen=True)
    class BadTrackingResult:
        frame_index: int
        timestamp: float
        tracks: tuple[BadTrack, ...]

    recognizer = SimpleNumberRecognizer()
    tracking_result = BadTrackingResult(
        frame_index=0,
        timestamp=0.0,
        tracks=(BadTrack(1),),
    )

    with pytest.raises(RecognitionInputError):
        recognizer.recognize(make_frame(), tracking_result)
