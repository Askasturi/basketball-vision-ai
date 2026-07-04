
import cv2
import numpy as np
import pytest


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory):
    """
    Create a temporary sample video for testing.
    """
    temp_dir = tmp_path_factory.mktemp("videos")
    video_path = temp_dir / "sample.mp4"

    width = 640
    height = 480
    fps = 30

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for _ in range(60):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        writer.write(frame)

    writer.release()

    return video_path
