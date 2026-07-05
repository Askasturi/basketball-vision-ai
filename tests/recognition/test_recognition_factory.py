from basketball_vision.recognition import (
    NumberRecognizerFactory,
    RecognitionConfig,
    SimpleNumberRecognizer,
)


def test_factory_creates_simple_recognizer():
    recognizer = NumberRecognizerFactory.create()

    assert isinstance(recognizer, SimpleNumberRecognizer)


def test_factory_passes_config():
    config = RecognitionConfig(track_id_to_number={1: "23"})
    recognizer = NumberRecognizerFactory.create(config=config)

    assert isinstance(recognizer, SimpleNumberRecognizer)
    assert recognizer.config is config
