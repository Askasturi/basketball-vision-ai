import pytest

from basketball_vision.pipeline import PipelineConfig


def test_pipeline_config_defaults() -> None:
    config = PipelineConfig()

    assert config.enable_tracking is True
    assert config.enable_classification is True
    assert config.enable_recognition is True
    assert config.require_detector is True
    assert config.detector_config is None
    assert config.tracker_config is None
    assert config.classifier_config is None
    assert config.recognizer_config is None


def test_pipeline_config_custom_values() -> None:
    config = PipelineConfig(
        enable_tracking=False,
        enable_classification=False,
        enable_recognition=False,
        require_detector=False,
        detector_config={"model": "test"},
    )

    assert config.enable_tracking is False
    assert config.enable_classification is False
    assert config.enable_recognition is False
    assert config.require_detector is False
    assert config.detector_config == {"model": "test"}


@pytest.mark.parametrize(
    "field_name",
    [
        "enable_tracking",
        "enable_classification",
        "enable_recognition",
        "require_detector",
    ],
)
def test_pipeline_config_boolean_fields_must_be_bool(field_name: str) -> None:
    kwargs = {field_name: "yes"}

    with pytest.raises(TypeError):
        PipelineConfig(**kwargs)
