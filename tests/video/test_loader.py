from pathlib import Path

import pytest

from basketball_vision.video.exceptions import VideoOpenError
from basketball_vision.video.loader import VideoLoader

TEST_VIDEO = Path("tests/assets/sample.mp4")


def test_invalid_path():
    with pytest.raises(VideoOpenError):
        VideoLoader("does_not_exist.mp4")


def test_loader_metadata():
    with VideoLoader(TEST_VIDEO) as loader:
        metadata = loader.metadata

        assert metadata.width > 0
        assert metadata.height > 0
        assert metadata.fps > 0
        assert metadata.frame_count > 0


def test_capture_is_open():
    with VideoLoader(TEST_VIDEO) as loader:
        assert loader.capture.isOpened()


def test_release():
    loader = VideoLoader(TEST_VIDEO)

    loader.release()

    assert not loader.capture.isOpened()
