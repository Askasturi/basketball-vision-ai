import pytest

from basketball_vision.classification import (
    BasePlayerClassifier,
    ClassifierType,
    ColorTeamClassifier,
    ColorTeamClassifierConfig,
    PlayerClassifierFactory,
)


def test_factory_creates_color_classifier_from_enum() -> None:
    classifier = PlayerClassifierFactory.create(ClassifierType.COLOR)

    assert isinstance(classifier, ColorTeamClassifier)


def test_factory_creates_color_classifier_from_string() -> None:
    classifier = PlayerClassifierFactory.create("color")

    assert isinstance(classifier, ColorTeamClassifier)


def test_factory_passes_config() -> None:
    config = ColorTeamClassifierConfig(team_a_color=(1, 2, 3))

    classifier = PlayerClassifierFactory.create(ClassifierType.COLOR, config)

    assert isinstance(classifier, ColorTeamClassifier)
    assert classifier.config == config


def test_factory_lists_available_classifiers() -> None:
    assert ClassifierType.COLOR in PlayerClassifierFactory.available_classifiers()


def test_factory_rejects_unknown_classifier_type() -> None:
    with pytest.raises(ValueError):
        PlayerClassifierFactory.create("unknown")


def test_factory_can_register_custom_classifier() -> None:
    class CustomClassifier(BasePlayerClassifier):
        def classify(self, frame, tracking_result):
            raise NotImplementedError

    PlayerClassifierFactory.register(ClassifierType.COLOR, CustomClassifier)

    try:
        classifier = PlayerClassifierFactory.create(ClassifierType.COLOR)
        assert isinstance(classifier, CustomClassifier)
    finally:
        PlayerClassifierFactory.register(
            ClassifierType.COLOR, ColorTeamClassifier)
