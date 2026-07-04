
import numpy as np
import pytest

from basketball_vision.video.exceptions import VideoWriteError
from basketball_vision.video.writer import VideoWriter


def test_create_writer(tmp_path):
    output = tmp_path / "output.mp4"

    with VideoWriter(
        output_path=output,
        width=640,
        height=480,
        fps=30,
    ):
        pass

    assert output.exists()


def test_write_frame(tmp_path):
    output = tmp_path / "video.mp4"

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with VideoWriter(
        output_path=output,
        width=640,
        height=480,
        fps=30,
    ) as writer:
        writer.write(frame)

    assert output.exists()


def test_none_frame_raises(tmp_path):
    output = tmp_path / "video.mp4"

    with VideoWriter(
        output_path=output,
        width=640,
        height=480,
        fps=30,
    ) as writer:
        with pytest.raises(VideoWriteError):
            writer.write(None)
