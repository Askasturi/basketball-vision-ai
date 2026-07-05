
"""Base interface for player number recognizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from basketball_vision.recognition.config import RecognitionConfig
from basketball_vision.recognition.number import RecognitionResult


class BaseNumberRecognizer(ABC):
    """Abstract base class for player number recognizers."""

    def __init__(self, config: RecognitionConfig | None = None) -> None:
        self.config = config or RecognitionConfig()

    @abstractmethod
    def recognize(
        self,
        frame: np.ndarray,
        tracking_result: Any,
        classification_result: Any | None = None,
    ) -> RecognitionResult:
        """Recognize jersey numbers from tracked players."""
