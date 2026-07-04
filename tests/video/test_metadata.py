from dataclasses import FrozenInstanceError

import pytest

from basketball_vision.video.metadata import VideoMetadata


def test_metadata_creation():
    metadata = VideoMetadata(
        width=1920,
        height=1080,
        fps=30.0,
        frame_count=900,
        codec="mp4v",
        duration_seconds=30.0,
    )

    assert metadata.width == 1920
    assert metadata.height == 1080
    assert metadata.fps == 30.0
    assert metadata.frame_count == 900
    assert metadata.codec == "mp4v"
    assert metadata.duration_seconds == 30.0


def test_resolution_property():
    metadata = VideoMetadata(
        width=1280,
        height=720,
        fps=60.0,
        frame_count=600,
        codec="avc1",
        duration_seconds=10.0,
    )

    assert metadata.resolution == (1280, 720)


def test_aspect_ratio():
    metadata = VideoMetadata(
        width=1920,
        height=1080,
        fps=30,
        frame_count=100,
        codec="mp4v",
        duration_seconds=5,
    )

    assert metadata.aspect_ratio == pytest.approx(16 / 9)


def test_metadata_is_immutable():
    metadata = VideoMetadata(
        width=640,
        height=480,
        fps=24,
        frame_count=240,
        codec="MJPG",
        duration_seconds=10,
    )

    with pytest.raises(FrozenInstanceError):
        metadata.width = 100
