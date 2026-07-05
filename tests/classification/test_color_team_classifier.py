from enum import Enum
from types import SimpleNamespace

import numpy as np
import pytest

from basketball_vision.classification.color_team_classifier import ColorTeamClassifier
from basketball_vision.classification.config import ColorTeamClassifierConfig
from basketball_vision.classification.types import TeamLabel


class TrackState(Enum):
    ACTIVE = "active"
    LOST = "lost"
    REMOVED = "removed"


def make_track(
    track_id: int,
    bounding_box: tuple[int, int, int, int],
    state: TrackState = TrackState.ACTIVE,
):
    detection = SimpleNamespace(bounding_box=bounding_box)
    return SimpleNamespace(track_id=track_id, detection=detection, state=state)


def make_tracking_result(*tracks):
    return SimpleNamespace(frame_index=5, timestamp=2.5, tracks=tracks)


def test_color_classifier_assigns_team_a() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10:40, 10:40] = (250, 5, 5)

    track = make_track(1, (10, 10, 40, 40))
    tracking_result = make_tracking_result(track)

    classifier = ColorTeamClassifier(
        ColorTeamClassifierConfig(
            team_a_color=(255, 0, 0),
            team_b_color=(0, 0, 255),
        )
    )

    result = classifier.classify(frame, tracking_result)

    assert result.frame_index == 5
    assert result.timestamp == 2.5
    assert len(result.assignments) == 1
    assert result.assignments[0].team == TeamLabel.TEAM_A
    assert result.assignments[0].track_id == 1


def test_color_classifier_assigns_team_b() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:60, 20:60] = (5, 5, 250)

    track = make_track(2, (20, 20, 60, 60))
    tracking_result = make_tracking_result(track)

    classifier = ColorTeamClassifier(
        ColorTeamClassifierConfig(
            team_a_color=(255, 0, 0),
            team_b_color=(0, 0, 255),
        )
    )

    result = classifier.classify(frame, tracking_result)

    assert result.assignments[0].team == TeamLabel.TEAM_B
    assert result.assignments[0].track_id == 2


def test_color_classifier_assigns_unknown_when_far_from_team_colors() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10:40, 10:40] = (0, 255, 0)

    track = make_track(1, (10, 10, 40, 40))
    tracking_result = make_tracking_result(track)

    classifier = ColorTeamClassifier(
        ColorTeamClassifierConfig(
            team_a_color=(255, 0, 0),
            team_b_color=(0, 0, 255),
            unknown_distance_threshold=80.0,
        )
    )

    result = classifier.classify(frame, tracking_result)

    assert result.assignments[0].team == TeamLabel.UNKNOWN


def test_color_classifier_ignores_lost_and_removed_tracks_by_default() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10:40, 10:40] = (255, 0, 0)

    lost_track = make_track(1, (10, 10, 40, 40), TrackState.LOST)
    removed_track = make_track(2, (10, 10, 40, 40), TrackState.REMOVED)
    tracking_result = make_tracking_result(lost_track, removed_track)

    classifier = ColorTeamClassifier()

    result = classifier.classify(frame, tracking_result)

    assert len(result.assignments) == 0


def test_color_classifier_can_include_lost_tracks() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10:40, 10:40] = (255, 0, 0)

    lost_track = make_track(1, (10, 10, 40, 40), TrackState.LOST)
    tracking_result = make_tracking_result(lost_track)

    classifier = ColorTeamClassifier(
        ColorTeamClassifierConfig(include_lost_tracks=True)
    )

    result = classifier.classify(frame, tracking_result)

    assert len(result.assignments) == 1
    assert result.assignments[0].team == TeamLabel.TEAM_A


def test_color_classifier_returns_unknown_for_invalid_crop() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    track = make_track(1, (10, 10, 10, 40))
    tracking_result = make_tracking_result(track)

    classifier = ColorTeamClassifier()

    result = classifier.classify(frame, tracking_result)

    assert result.assignments[0].team == TeamLabel.UNKNOWN
    assert result.assignments[0].confidence == 0.0


def test_color_classifier_supports_direct_track_bounding_box() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10:40, 10:40] = (255, 0, 0)

    track = SimpleNamespace(
        track_id=1,
        bounding_box=(10, 10, 40, 40),
        state=TrackState.ACTIVE,
    )
    tracking_result = make_tracking_result(track)

    classifier = ColorTeamClassifier()

    result = classifier.classify(frame, tracking_result)

    assert result.assignments[0].team == TeamLabel.TEAM_A


def test_color_classifier_rejects_invalid_frame_type() -> None:
    classifier = ColorTeamClassifier()

    with pytest.raises(TypeError):
        classifier.classify("not an image", make_tracking_result())


def test_color_classifier_rejects_invalid_frame_shape() -> None:
    classifier = ColorTeamClassifier()

    with pytest.raises(ValueError):
        classifier.classify(
            np.zeros((100, 100), dtype=np.uint8), make_tracking_result())
