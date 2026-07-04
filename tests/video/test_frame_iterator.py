from basketball_vision.video.frame import VideoFrame
from basketball_vision.video.frame_iterator import FrameIterator
from basketball_vision.video.loader import VideoLoader


def test_iterator_returns_video_frame():
    with VideoLoader("tests/assets/sample.mp4") as loader:
        iterator = FrameIterator(loader)

        frame = next(iterator)

        assert isinstance(frame, VideoFrame)


def test_first_frame_index():
    with VideoLoader("tests/assets/sample.mp4") as loader:
        iterator = FrameIterator(loader)

        frame = next(iterator)

        assert frame.index == 0


def test_second_frame_index():
    with VideoLoader("tests/assets/sample.mp4") as loader:
        iterator = FrameIterator(loader)

        next(iterator)

        frame = next(iterator)

        assert frame.index == 1


def test_timestamp_increases():
    with VideoLoader("tests/assets/sample.mp4") as loader:
        iterator = FrameIterator(loader)

        frame1 = next(iterator)
        frame2 = next(iterator)

        assert frame2.timestamp > frame1.timestamp


def test_image_exists():
    with VideoLoader("tests/assets/sample.mp4") as loader:
        iterator = FrameIterator(loader)

        frame = next(iterator)

        assert frame.image is not None
