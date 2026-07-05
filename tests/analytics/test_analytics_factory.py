from basketball_vision.analytics import (
    AnalyticsConfig,
    AnalyticsFactory,
    GameStatisticsAccumulator,
)


def test_factory_creates_game_accumulator() -> None:
    accumulator = AnalyticsFactory.create_game_accumulator(
        AnalyticsConfig(fps=24.0),
    )

    assert isinstance(accumulator, GameStatisticsAccumulator)
    assert accumulator.fps == 24.0
