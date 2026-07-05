"""Factory helpers for visualization renderers."""

from __future__ import annotations

from basketball_vision.visualization.config import VisualizationConfig
from basketball_vision.visualization.renderer import Renderer


class VisualizationFactory:
    """Factory for creating visualization renderers."""

    @staticmethod
    def create_default() -> Renderer:
        """Create a renderer with default visualization settings."""
        return Renderer(config=VisualizationConfig())

    @staticmethod
    def create(config: VisualizationConfig) -> Renderer:
        """Create a renderer with the provided visualization settings."""
        return Renderer(config=config)
