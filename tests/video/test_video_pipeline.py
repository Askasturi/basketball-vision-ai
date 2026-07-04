from basketball_vision.video import (
    FrameIterator,
    VideoLoader,
    VideoWriter,
)


def test_video_pipeline(tmp_path):
    input_video = "tests/assets/sample.mp4"
    output_video = tmp_path / "copy.mp4"

    with VideoLoader(input_video) as loader:
        metadata = loader.metadata

        with VideoWriter(
            output_path=output_video,
            width=metadata.width,
            height=metadata.height,
            fps=metadata.fps,
        ) as writer:
            for frame in FrameIterator(loader):
                writer.write(frame.image)

    assert output_video.exists()

    with VideoLoader(output_video) as loader:
        assert loader.metadata.frame_count > 0
