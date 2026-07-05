
"""Factory for player number recognizers."""

from __future__ import annotations

from basketball_vision.recognition.base_recognizer import BaseNumberRecognizer
from basketball_vision.recognition.config import RecognitionConfig
from basketball_vision.recognition.simple_number_recognizer import (
    SimpleNumberRecognizer,
)
from basketball_vision.recognition.types import NumberRecognizerType


class NumberRecognizerFactory:
    """Factory for creating number recognizers."""

    @staticmethod
    def create(config: RecognitionConfig | None = None) -> BaseNumberRecognizer:
        """Create a number recognizer from config."""
        config = config or RecognitionConfig()

        if config.recognizer_type is NumberRecognizerType.SIMPLE:
            return SimpleNumberRecognizer(config=config)

        msg = f"Unsupported recognizer type: {config.recognizer_type}"
        raise ValueError(msg)
