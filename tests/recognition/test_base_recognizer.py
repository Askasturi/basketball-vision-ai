
import pytest

from basketball_vision.recognition import BaseNumberRecognizer, RecognitionConfig


def test_base_number_recognizer_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseNumberRecognizer()


def test_base_number_recognizer_stores_config():
    class ConcreteRecognizer(BaseNumberRecognizer):
        def recognize(self, frame, tracking_result, classification_result=None):
            return None

    config = RecognitionConfig(confidence_threshold=0.8)
    recognizer = ConcreteRecognizer(config=config)

    assert recognizer.config is config
