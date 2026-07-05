"""Tests for the Basketball Vision CLI."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from basketball_vision.cli.main import main


def test_process_video_missing_input_returns_error(tmp_path):
    missing_video = tmp_path / "missing.mp4"

    exit_code = main(["process-video", str(missing_video)])

    assert exit_code == 1


def test_process_video_runs_pipeline_and_exports_json(tmp_path):
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"fake video")

    output_dir = tmp_path / "outputs"

    mock_pipeline = MagicMock()
    mock_pipeline.process_video.return_value = {
        "frames": [{"frame_index": 0, "detections": []}]
    }

    with patch(
        "basketball_vision.cli.main.BasketballVisionPipeline",
        return_value=mock_pipeline,
    ):
        exit_code = main(
            [
                "process-video",
                str(input_video),
                "--output-dir",
                str(output_dir),
            ]
        )

    assert exit_code == 0
    assert (output_dir / "results.json").exists()
    mock_pipeline.process_video.assert_called_once()


def test_process_video_custom_json_output(tmp_path):
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"fake video")

    json_output = tmp_path / "custom" / "result.json"

    mock_pipeline = MagicMock()
    mock_pipeline.process_video.return_value = {"frames": []}

    with patch(
        "basketball_vision.cli.main.BasketballVisionPipeline",
        return_value=mock_pipeline,
    ):
        exit_code = main(
            [
                "process-video",
                str(input_video),
                "--json-output",
                str(json_output),
            ]
        )

    assert exit_code == 0
    assert json_output.exists()
