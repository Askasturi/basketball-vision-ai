from dataclasses import dataclass

import numpy as np

from basketball_vision.analytics import (
    AnalyticsResult,
    GameStatistics,
)
from basketball_vision.visualization.renderer import Renderer


@dataclass(frozen=True)
class FakePipelineResult:
    frame_index: int
    analytics: AnalyticsResult


def make_analytics_result() -> AnalyticsResult:
    game = GameStatistics(
        total_frames=300,
        video_seconds=10.0,
        number_of_players=10,
        average_players_on_court=10.0,
        ball_detections=25,
        team_counts={"blue": 5, "white": 5},
        estimated_ball_possession={"blue": 0.5, "white": 0.5},
        detection_counts={"player": 3000, "ball": 25},
    )
    return AnalyticsResult(players=(), game=game)


def test_renderer_draws_analytics_overlay() -> None:
    renderer = Renderer()
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    result = FakePipelineResult(
        frame_index=245,
        analytics=make_analytics_result(),
    )

    rendered = renderer.render(frame=frame, pipeline_frame_result=result)

    assert rendered.shape == frame.shape
    assert np.any(rendered != frame)


def test_renderer_without_analytics_overlay_still_works() -> None:
    renderer = Renderer()
    frame = np.zeros((160, 240, 3), dtype=np.uint8)

    rendered = renderer.render(frame=frame, pipeline_frame_result=None)

    assert np.array_equal(rendered, frame)
