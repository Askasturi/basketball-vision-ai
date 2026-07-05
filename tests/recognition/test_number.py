
from dataclasses import dataclass

import pytest

from basketball_vision.recognition import (
    NumberRecognitionStatus,
    PlayerNumberRecognition,
    RecognitionConfigurationError,
    RecognitionResult,
)


@dataclass(frozen=True)
class DummyTrack:
    track_id: int


def test_player_number_recognition_recognized():
    track = DummyTrack(track_id=1)

    result = PlayerNumberRecognition(
        track_id=1,
        number="23",
        confidence=0.9,
        status=NumberRecognitionStatus.RECOGNIZED,
        track=track,
    )

    assert result.track_id == 1
    assert result.number == "23"
    assert result.confidence == 0.9
    assert result.status is NumberRecognitionStatus.RECOGNIZED
    assert result.track is track


def test_player_number_recognition_unknown():
    track = DummyTrack(track_id=1)

    result = PlayerNumberRecognition(
        track_id=1,
        number=None,
        confidence=0.0,
        status=NumberRecognitionStatus.UNKNOWN,
        track=track,
    )

    assert result.number is None
    assert result.status is NumberRecognitionStatus.UNKNOWN


def test_recognized_status_requires_number():
    track = DummyTrack(track_id=1)

    with pytest.raises(RecognitionConfigurationError):
        PlayerNumberRecognition(
            track_id=1,
            number=None,
            confidence=0.9,
            status=NumberRecognitionStatus.RECOGNIZED,
            track=track,
        )


def test_invalid_track_id():
    track = DummyTrack(track_id=-1)

    with pytest.raises(RecognitionConfigurationError):
        PlayerNumberRecognition(
            track_id=-1,
            number="23",
            confidence=0.9,
            status=NumberRecognitionStatus.RECOGNIZED,
            track=track,
        )


def test_invalid_confidence():
    track = DummyTrack(track_id=1)

    with pytest.raises(RecognitionConfigurationError):
        PlayerNumberRecognition(
            track_id=1,
            number="23",
            confidence=1.5,
            status=NumberRecognitionStatus.RECOGNIZED,
            track=track,
        )


def test_invalid_number():
    track = DummyTrack(track_id=1)

    with pytest.raises(RecognitionConfigurationError):
        PlayerNumberRecognition(
            track_id=1,
            number="A3",
            confidence=0.9,
            status=NumberRecognitionStatus.RECOGNIZED,
            track=track,
        )


def test_recognition_result():
    track = DummyTrack(track_id=1)
    recognition = PlayerNumberRecognition(
        track_id=1,
        number="23",
        confidence=1.0,
        status=NumberRecognitionStatus.RECOGNIZED,
        track=track,
    )

    result = RecognitionResult(
        frame_index=5,
        timestamp=1.25,
        recognitions=(recognition,),
    )

    assert result.frame_index == 5
    assert result.timestamp == 1.25
    assert result.recognitions == (recognition,)


def test_invalid_recognition_result_frame_index():
    with pytest.raises(RecognitionConfigurationError):
        RecognitionResult(frame_index=-1, timestamp=None, recognitions=())


def test_invalid_recognition_result_timestamp():
    with pytest.raises(RecognitionConfigurationError):
        RecognitionResult(frame_index=0, timestamp=-1.0, recognitions=())
