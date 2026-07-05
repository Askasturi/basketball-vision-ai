"""Factory for analytics components."""

from basketball_vision.analytics.analytics_engine import AnalyticsEngine
from basketball_vision.analytics.config import AnalyticsConfig
from basketball_vision.analytics.exporters import AnalyticsExporter
from basketball_vision.analytics.game_statistics import GameStatisticsAccumulator


class AnalyticsFactory:
    """Create analytics components."""

    @staticmethod
    def create_engine(config: AnalyticsConfig | None = None) -> AnalyticsEngine:
        """Create an analytics engine."""
        return AnalyticsEngine(config=config)

    @staticmethod
    def create_exporter(
        config: AnalyticsConfig | None = None,
    ) -> AnalyticsExporter:
        """Create an analytics exporter."""
        analytics_config = config or AnalyticsConfig()
        return AnalyticsExporter(output_dir=analytics_config.output_dir)

    @staticmethod
    def create_game_accumulator(
        config: AnalyticsConfig | None = None,
    ) -> GameStatisticsAccumulator:
        """Create a game statistics accumulator."""
        analytics_config = config or AnalyticsConfig()
        return GameStatisticsAccumulator(fps=analytics_config.fps)
