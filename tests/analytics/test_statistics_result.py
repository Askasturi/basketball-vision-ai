from dataclasses import FrozenInstanceError

import pytest

from basketball_vision.analytics import (
    AnalyticsResult,
    GameStatistics,
    PlayerStatistics,
)


def test_player_statistics_is_immutable() -> None:
    stats = PlayerStatistics(
        track_id=1,
        team="blue",
        jersey_number="23",
        frames_seen=10,
        first_frame=0,
        last_frame=9,
        time_visible=1.0,
        average_detection_confidence=0.9,
        average_bounding_box_area=100.0,
        distance_travelled_pixels=50.0,
        average_speed=50.0,
        maximum_speed=75.0,
    )

    with pytest.raises(FrozenInstanceError):
        stats.track_id = 2


def test_analytics_result_holds_players_and_game() -> None:
    player = PlayerStatistics(
        track_id=1,
        team="blue",
        jersey_number="23",
        frames_seen=10,
        first_frame=0,
        last_frame=9,
        time_visible=1.0,
        average_detection_confidence=0.9,
        average_bounding_box_area=100.0,
        distance_travelled_pixels=50.0,
        average_speed=50.0,
        maximum_speed=75.0,
    )
    game = GameStatistics(
        total_frames=10,
        video_seconds=1.0,
        number_of_players=1,
        average_players_on_court=1.0,
        ball_detections=0,
        team_counts={"blue": 10},
        estimated_ball_possession={"blue": 1.0},
        detection_counts={"player": 10},
    )

    result = AnalyticsResult(players=(player,), game=game)

    assert result.players == (player,)
    assert result.game == game
