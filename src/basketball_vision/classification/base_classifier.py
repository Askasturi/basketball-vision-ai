"""Base classifier interface."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from basketball_vision.classification.config import ClassificationConfig
from basketball_vision.classification.player_team import ClassificationResult


class BasePlayerClassifier(ABC):
    """Abstract base class for player team classifiers."""

    def __init__(self, config: ClassificationConfig | None = None) -> None:
        self.config = config or ClassificationConfig()

    @abstractmethod
    def classify(self, frame: np.ndarray, tracking_result: Any) -> ClassificationResult:
        """Classify tracked players in a frame."""
