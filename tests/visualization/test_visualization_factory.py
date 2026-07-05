"""Tests for visualization factory."""

from __future__ import annotations

from basketball_vision.visualization import (
    Renderer,
    VisualizationConfig,
    VisualizationFactory,
)


def test_create_default_renderer() -> None:
    renderer = VisualizationFactory.create_default()

    assert isinstance(renderer, Renderer)
    assert isinstance(renderer.config, VisualizationConfig)


def test_create_renderer_with_config() -> None:
    config = VisualizationConfig(show_confidence=True, line_thickness=3)

    renderer = VisualizationFactory.create(config)

    assert isinstance(renderer, Renderer)
    assert renderer.config == config
