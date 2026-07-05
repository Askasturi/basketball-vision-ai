"""Tests for visualization renderer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from basketball_vision.visualization import Renderer, VisualizationConfig


@dataclass(frozen=True)
class MockObject:
    box: tuple[int, int, int, int]
    track_id: int
    team: str
    jersey_number: int
    confidence: float


@dataclass(frozen=True)
class MockFrameResult:
    frame_index: int
    objects: list[MockObject]


def test_renderer_preserves_shape() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    renderer = Renderer()

    rendered = renderer.render(frame, None)

    assert rendered.shape == frame.shape


def test_renderer_returns_copy() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    renderer = Renderer()

    rendered = renderer.render(frame, None)

    assert rendered is not frame


def test_renderer_does_not_modify_input_frame() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    original = frame.copy()
    result = MockFrameResult(
        frame_index=1,
        objects=[
            MockObject(
                box=(10, 10, 70, 90),
                track_id=7,
                team="TEAM_A",
                jersey_number=23,
                confidence=0.91,
            )
        ],
    )
    renderer = Renderer()

    _ = renderer.render(frame, result)

    assert np.array_equal(frame, original)


def test_renderer_draws_annotations() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    result = MockFrameResult(
        frame_index=1,
        objects=[
            MockObject(
                box=(10, 10, 70, 90),
                track_id=7,
                team="TEAM_A",
                jersey_number=23,
                confidence=0.91,
            )
        ],
    )
    renderer = Renderer()

    rendered = renderer.render(frame, result)

    assert np.any(rendered != frame)


def test_renderer_can_hide_optional_annotations() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    result = MockFrameResult(
        frame_index=1,
        objects=[
            MockObject(
                box=(10, 10, 70, 90),
                track_id=7,
                team="TEAM_A",
                jersey_number=23,
                confidence=0.91,
            )
        ],
    )
    config = VisualizationConfig(
        show_ids=False,
        show_team=False,
        show_numbers=False,
        show_confidence=False,
        show_frame_index=False,
    )
    renderer = Renderer(config=config)

    rendered = renderer.render(frame, result)

    assert np.any(rendered != frame)


def test_renderer_supports_dictionary_objects() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    result = {
        "frame_index": 2,
        "objects": [
            {
                "box": (20, 20, 80, 95),
                "track_id": 4,
                "team": "TEAM_B",
                "number": 11,
                "confidence": 0.88,
            }
        ],
    }
    renderer = Renderer(VisualizationConfig(show_confidence=True))

    rendered = renderer.render(frame, result)

    assert np.any(rendered != frame)
