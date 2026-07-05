from types import SimpleNamespace

from basketball_vision.classification.player_team import (
    ClassificationResult,
    PlayerTeamAssignment,
)
from basketball_vision.classification.types import TeamLabel


def test_player_team_assignment() -> None:
    track = SimpleNamespace(track_id=1)

    assignment = PlayerTeamAssignment(
        track_id=1,
        team=TeamLabel.TEAM_A,
        confidence=0.95,
        track=track,
    )

    assert assignment.track_id == 1
    assert assignment.team == TeamLabel.TEAM_A
    assert assignment.confidence == 0.95
    assert assignment.track is track


def test_classification_result_len_iter_and_lookup() -> None:
    track = SimpleNamespace(track_id=7)
    assignment = PlayerTeamAssignment(
        track_id=7,
        team=TeamLabel.TEAM_B,
        confidence=0.9,
        track=track,
    )

    result = ClassificationResult(
        frame_index=10,
        timestamp=1.5,
        assignments=(assignment,),
    )

    assert len(result) == 1
    assert list(result) == [assignment]
    assert result.get_assignment(7) == assignment
    assert result.get_assignment(99) is None
