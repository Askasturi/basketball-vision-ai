import csv
import json
from pathlib import Path

from basketball_vision.analytics import (
    PLAYER_STATISTICS_COLUMNS,
    AnalyticsExporter,
    AnalyticsFactory,
    AnalyticsResult,
    GameStatistics,
    PlayerStatistics,
)


def make_result() -> AnalyticsResult:
    player = PlayerStatistics(
        track_id=4,
        team="blue",
        jersey_number="23",
        frames_seen=481,
        first_frame=0,
        last_frame=480,
        time_visible=16.0333333333,
        average_detection_confidence=0.91234,
        average_bounding_box_area=1200.0,
        distance_travelled_pixels=5210.4,
        average_speed=325.0,
        maximum_speed=700.0,
    )
    game = GameStatistics(
        total_frames=540,
        video_seconds=18.0,
        number_of_players=1,
        average_players_on_court=1.0,
        ball_detections=12,
        team_counts={"blue": 481},
        estimated_ball_possession={"blue": 1.0},
        detection_counts={"player": 481, "ball": 12},
    )
    return AnalyticsResult(players=(player,), game=game)


def test_export_player_statistics_csv(tmp_path: Path) -> None:
    exporter = AnalyticsExporter(output_dir=tmp_path)
    path = exporter.export_player_statistics(make_result())

    assert path == tmp_path / "player_statistics.csv"
    assert path.exists()

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows[0].keys() == set(PLAYER_STATISTICS_COLUMNS)
    assert rows[0]["Track ID"] == "4"
    assert rows[0]["Team"] == "blue"
    assert rows[0]["Jersey Number"] == "23"
    assert rows[0]["Frames Seen"] == "481"
    assert rows[0]["Distance"] == "5210.400"
    assert rows[0]["Average Speed"] == "325.000"
    assert rows[0]["Average Confidence"] == "0.912"


def test_export_game_statistics_json(tmp_path: Path) -> None:
    exporter = AnalyticsExporter(output_dir=tmp_path)
    path = exporter.export_game_statistics(make_result())

    assert path == tmp_path / "game_statistics.json"
    assert path.exists()

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["frames"] == 540
    assert payload["video_seconds"] == 18.0
    assert payload["number_of_players"] == 1
    assert payload["ball_detections"] == 12
    assert payload["team_counts"] == {"blue": 481}
    assert payload["detection_counts"] == {"player": 481, "ball": 12}
    assert payload["players"][0]["track_id"] == 4
    assert payload["players"][0]["team"] == "blue"
    assert payload["players"][0]["jersey_number"] == "23"
    assert payload["players"][0]["frames_seen"] == 481
    assert payload["players"][0]["distance_pixels"] == 5210.4


def test_export_all_creates_csv_and_json(tmp_path: Path) -> None:
    exporter = AnalyticsExporter(output_dir=tmp_path)

    player_path, game_path = exporter.export_all(make_result())

    assert player_path.exists()
    assert game_path.exists()


def test_to_json_payload_is_serializable() -> None:
    payload = AnalyticsExporter.to_json_payload(make_result())

    encoded = json.dumps(payload)

    assert "players" in encoded
    assert "distance_pixels" in encoded


def test_factory_creates_exporter(tmp_path: Path) -> None:
    config = AnalyticsFactory.create_exporter

    exporter = config()

    assert isinstance(exporter, AnalyticsExporter)
