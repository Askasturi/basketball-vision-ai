import pytest

from basketball_vision.classification.config import (
    ClassificationConfig,
    ColorTeamClassifierConfig,
)
from basketball_vision.classification.exceptions import InvalidClassificationConfigError


def test_base_config_defaults() -> None:
    config = ClassificationConfig()

    assert config.min_confidence == 0.0
    assert config.include_lost_tracks is False
    assert config.include_removed_tracks is False


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_base_config_rejects_invalid_min_confidence(value: float) -> None:
    with pytest.raises(InvalidClassificationConfigError):
        ClassificationConfig(min_confidence=value)


def test_color_config_defaults() -> None:
    config = ColorTeamClassifierConfig()

    assert config.team_a_color == (255, 0, 0)
    assert config.team_b_color == (0, 0, 255)
    assert config.unknown_distance_threshold == 140.0


@pytest.mark.parametrize(
    "color",
    [
        (255, 0),
        (255, 0, 0, 0),
        (-1, 0, 0),
        (256, 0, 0),
        (255.0, 0, 0),
    ],
)
def test_color_config_rejects_invalid_color(color) -> None:
    with pytest.raises(InvalidClassificationConfigError):
        ColorTeamClassifierConfig(team_a_color=color)


def test_color_config_rejects_negative_threshold() -> None:
    with pytest.raises(InvalidClassificationConfigError):
        ColorTeamClassifierConfig(unknown_distance_threshold=-1)


def test_color_config_rejects_negative_padding() -> None:
    with pytest.raises(InvalidClassificationConfigError):
        ColorTeamClassifierConfig(crop_padding=-1)


def test_color_config_rejects_non_positive_min_crop_area() -> None:
    with pytest.raises(InvalidClassificationConfigError):
        ColorTeamClassifierConfig(min_crop_area=0)
