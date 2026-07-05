from pathlib import Path

import pytest

from basketball_vision.analytics import AnalyticsConfig, AnalyticsConfigurationError


def test_default_config() -> None:
    config = AnalyticsConfig()

    assert config.fps == 30.0
    assert config.output_dir == Path("outputs/analytics")
    assert config.export_csv is True
    assert config.export_json is True


def test_output_dir_string_is_converted_to_path() -> None:
    config = AnalyticsConfig(output_dir="custom/analytics")

    assert config.output_dir == Path("custom/analytics")


def test_invalid_fps_raises() -> None:
    with pytest.raises(AnalyticsConfigurationError):
        AnalyticsConfig(fps=0)
