from typing import Any

import numpy as np

from basketball_vision.classification.base_classifier import BasePlayerClassifier
from basketball_vision.classification.config import ClassificationConfig
from basketball_vision.classification.player_team import ClassificationResult


class DummyClassifier(BasePlayerClassifier):
    def classify(self, frame: np.ndarray, tracking_result: Any) -> ClassificationResult:
        return ClassificationResult(frame_index=0, timestamp=None, assignments=())


def test_base_classifier_stores_config() -> None:
    config = ClassificationConfig(min_confidence=0.5)

    classifier = DummyClassifier(config)

    assert classifier.config == config


def test_base_classifier_uses_default_config() -> None:
    classifier = DummyClassifier()

    assert isinstance(classifier.config, ClassificationConfig)
