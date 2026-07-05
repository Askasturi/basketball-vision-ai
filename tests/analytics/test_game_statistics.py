import pytest

from basketball_vision.analytics import GameStatisticsAccumulator


def test_game_statistics_accumulator_tracks_game_metrics() -> None:
    accumulator = GameStatisticsAccumulator(fps=30.0)

    accumulator.update(
        players_on_court=2,
        teams=["blue", "white"],
        detection_labels=["player", "player", "ball"],
        ball_detections=1,
    )
    accumulator.update(
        players_on_court=1,
        teams=["blue"],
        detection_labels=["player"],
        ball_detections=0,
    )

    result = accumulator.to_result(number_of_players=2)

    assert result.total_frames == 2
    assert result.video_seconds == pytest.approx(2 / 30)
    assert result.number_of_players == 2
    assert result.average_players_on_court == pytest.approx(1.5)
    assert result.ball_detections == 1
    assert result.team_counts == {"blue": 2, "white": 1}
    assert result.estimated_ball_possession["blue"] == pytest.approx(2 / 3)
    assert result.estimated_ball_possession["white"] == pytest.approx(1 / 3)
    assert result.detection_counts == {"player": 3, "ball": 1}
