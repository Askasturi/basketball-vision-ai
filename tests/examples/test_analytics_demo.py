import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def load_analytics_demo() -> ModuleType:
    module_path = Path("examples/analytics_demo.py")
    spec = importlib.util.spec_from_file_location(
        "analytics_demo", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules["analytics_demo"] = module
    spec.loader.exec_module(module)
    return module


def test_analytics_demo_creates_video_csv_and_json(tmp_path: Path) -> None:
    analytics_demo = load_analytics_demo()

    output_video = tmp_path / "analytics_demo.mp4"
    analytics_dir = tmp_path / "analytics"

    video_path, player_csv, game_json = analytics_demo.run_demo(
        output_video=output_video,
        analytics_dir=analytics_dir,
        frames=5,
        fps=10.0,
    )

    assert video_path.exists()
    assert video_path.stat().st_size > 0

    assert player_csv.exists()
    assert "Track ID" in player_csv.read_text(encoding="utf-8")

    assert game_json.exists()
    payload = json.loads(game_json.read_text(encoding="utf-8"))

    assert payload["frames"] == 5
    assert payload["video_seconds"] == 0.5
    assert payload["number_of_players"] == 2
    assert len(payload["players"]) == 2
