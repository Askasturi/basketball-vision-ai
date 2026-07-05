
import importlib.util
import sys
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[2] / "examples" / "full_pipeline_demo.py"
SPEC = importlib.util.spec_from_file_location("full_pipeline_demo", MODULE_PATH)
assert SPEC is not None
full_pipeline_demo = importlib.util.module_from_spec(SPEC)
sys.modules["full_pipeline_demo"] = full_pipeline_demo
assert SPEC.loader is not None
SPEC.loader.exec_module(full_pipeline_demo)


def test_create_demo_frame_returns_valid_numpy_frame() -> None:
    frame = full_pipeline_demo.create_demo_frame(frame_index=0, width=320, height=180)

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (180, 320, 3)
    assert frame.dtype == np.uint8


def test_mock_pipeline_result_has_expected_structure() -> None:
    result = full_pipeline_demo.create_mock_pipeline_result(frame_index=3)

    assert result["frame_index"] == 3
    assert "objects" in result
    assert len(result["objects"]) > 0
    assert all(
        isinstance(obj, full_pipeline_demo.DemoObject)
        for obj in result["objects"]
    )


def test_ensure_output_dir_creates_parent_directory(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "demo.mp4"

    returned_path = full_pipeline_demo.ensure_output_dir(output_path)

    assert returned_path == output_path
    assert output_path.parent.exists()


def test_write_demo_video_does_not_require_real_yolo_weights(tmp_path: Path) -> None:
    output_path = tmp_path / "full_pipeline_demo.mp4"

    result_path = full_pipeline_demo.write_demo_video(
        output_path=output_path,
        frame_count=3,
        fps=6,
        width=320,
        height=180,
    )

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
