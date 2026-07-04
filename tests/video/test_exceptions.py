import pytest

from basketball_vision.video.exceptions import (
    InvalidFrameError,
    UnsupportedCodecError,
    VideoOpenError,
    VideoWriteError,
)


def test_video_open_error():
    with pytest.raises(VideoOpenError):
        raise VideoOpenError("Cannot open file")


def test_invalid_frame_error():
    with pytest.raises(InvalidFrameError):
        raise InvalidFrameError("Invalid frame")


def test_codec_error():
    with pytest.raises(UnsupportedCodecError):
        raise UnsupportedCodecError("Unsupported codec")


def test_writer_error():
    with pytest.raises(VideoWriteError):
        raise VideoWriteError("Write failed")
