
import pytest

from basketball_vision.recognition import (
    NumberRecognizerType,
    RecognitionConfig,
    RecognitionConfigurationError,
)


def test_default_config():
    config = RecognitionConfig()

    assert config.recognizer_type is NumberRecognizerType.SIMPLE
    assert config.confidence_threshold == 0.50
    assert config.include_lost_tracks is False
    assert config.include_removed_tracks is False
    assert config.valid_min_number == 0
    assert config.valid_max_number == 99
    assert config.track_id_to_number == {}


def test_custom_config():
    config = RecognitionConfig(
        confidence_threshold=0.75,
        include_lost_tracks=True,
        include_removed_tracks=True,
        valid_min_number=1,
        valid_max_number=55,
        track_id_to_number={1: "23"},
    )

    assert config.confidence_threshold == 0.75
    assert config.include_lost_tracks is True
    assert config.include_removed_tracks is True
    assert config.valid_min_number == 1
    assert config.valid_max_number == 55
    assert config.track_id_to_number == {1: "23"}


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_invalid_confidence_threshold(value):
    with pytest.raises(RecognitionConfigurationError):
        RecognitionConfig(confidence_threshold=value)


def test_invalid_number_range():
    with pytest.raises(RecognitionConfigurationError):
        RecognitionConfig(valid_min_number=10, valid_max_number=5)


def test_negative_min_number():
    with pytest.raises(RecognitionConfigurationError):
        RecognitionConfig(valid_min_number=-1)


def test_invalid_track_id_key():
    with pytest.raises(RecognitionConfigurationError):
        RecognitionConfig(track_id_to_number={"1": "23"})  # type: ignore[dict-item]


def test_negative_track_id_key():
    with pytest.raises(RecognitionConfigurationError):
        RecognitionConfig(track_id_to_number={-1: "23"})


def test_non_digit_number():
    with pytest.raises(RecognitionConfigurationError):
        RecognitionConfig(track_id_to_number={1: "A3"})


def test_number_out_of_range():
    with pytest.raises(RecognitionConfigurationError):
        RecognitionConfig(
            valid_min_number=0,
            valid_max_number=50,
            track_id_to_number={1: "99"},
        )
