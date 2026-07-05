"""Tests for renderer support across result object shapes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from basketball_vision.visualization import Renderer, VisualizationConfig


@dataclass(frozen=True)
class MockBoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class ObjectWithBoundingBox:
    bounding_box: MockBoundingBox
    id: int
    team: str
    number: int
    confidence: float


@dataclass(frozen=True)
class ResultWithResults:
    frame_index: int
    results: list[ObjectWithBoundingBox]


@dataclass(frozen=True)
class ObjectWithBbox:
    bbox: tuple[int, int, int, int]
    track_id: int
    team: str


@dataclass(frozen=True)
class ResultWithDetections:
    frame_index: int
    detections: list[ObjectWithBbox]


def test_renderer_supports_bounding_box_object() -> None:
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    result = ResultWithResults(
        frame_index=3,
        results=[
            ObjectWithBoundingBox(
                bounding_box=MockBoundingBox(10, 15, 80, 100),
                id=12,
                team="home",
                number=30,
                confidence=0.97,
            )
        ],
    )
    renderer = Renderer(VisualizationConfig(show_confidence=True))

    rendered = renderer.render(frame, result)

    assert rendered.shape == frame.shape
    assert np.any(rendered != frame)


def test_renderer_supports_bbox_attribute() -> None:
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    result = ResultWithDetections(
        frame_index=4,
        detections=[
            ObjectWithBbox(
                bbox=(20, 25, 90, 110),
                track_id=3,
                team="away",
            )
        ],
    )
    renderer = Renderer()

    rendered = renderer.render(frame, result)

    assert rendered.shape == frame.shape
    assert np.any(rendered != frame)


def test_renderer_handles_objects_without_boxes() -> None:
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    result = {
        "frame_index": 5,
        "objects": [
            {
                "track_id": 1,
                "team": "TEAM_A",
            }
        ],
    }
    renderer = Renderer(VisualizationConfig(show_frame_index=False))

    rendered = renderer.render(frame, result)

    assert rendered.shape == frame.shape
    assert np.array_equal(rendered, frame)
