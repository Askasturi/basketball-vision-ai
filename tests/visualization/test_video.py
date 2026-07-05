"""Tests for annotated video writing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from basketball_vision.visualization import write_annotated_video


def test_write_annotated_video_creates_output_file(tmp_path: Path) -> None:
    frames = [
        np.zeros((64, 64, 3), dtype=np.uint8),
        np.zeros((64, 64, 3), dtype=np.uint8),
    ]
    results = [
        {
            "frame_index": 0,
            "objects": [
                {
                    "box": (10, 10, 40, 50),
                    "track_id": 1,
                    "team": "TEAM_A",
                    "number": 23,
                }
            ],
        },
        {
            "frame_index": 1,
            "objects": [
                {
                    "box": (12, 10, 42, 50),
                    "track_id": 1,
                    "team": "TEAM_A",
                    "number": 23,
                }
            ],
        },
    ]
    output_path = tmp_path / "annotated.mp4"

    returned_path = write_annotated_video(
        frames=frames,
        results=results,
        output_path=output_path,
        fps=24.0,
    )

    assert returned_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_write_annotated_video_creates_parent_directory(tmp_path: Path) -> None:
    frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
    results = [{"frame_index": 0, "objects": []}]
    output_path = tmp_path / "nested" / "annotated.mp4"

    write_annotated_video(
        frames=frames,
        results=results,
        output_path=output_path,
        fps=24.0,
    )

    assert output_path.exists()
