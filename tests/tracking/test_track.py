from basketball_vision.detection import BoundingBox, Detection
from basketball_vision.tracking import Track, TrackingResult, TrackState


def make_detection(x_offset: float = 0.0) -> Detection:
    return Detection(
        bounding_box=BoundingBox(
            x1=10.0 + x_offset,
            y1=20.0,
            x2=50.0 + x_offset,
            y2=80.0,
        ),
        confidence=0.9,
        class_id=0,
        class_name="player",
    )


def test_track_defaults() -> None:
    detection = make_detection()
    track = Track(track_id=1, detection=detection)

    assert track.track_id == 1
    assert track.detection == detection
    assert track.state == TrackState.NEW
    assert track.age == 1
    assert track.hits == 1
    assert track.missed_frames == 0


def test_track_mark_updated() -> None:
    track = Track(track_id=1, detection=make_detection())
    updated = track.mark_updated(make_detection(x_offset=2.0))

    assert updated.track_id == 1
    assert updated.state == TrackState.ACTIVE
    assert updated.age == 2
    assert updated.hits == 2
    assert updated.missed_frames == 0


def test_track_mark_missed() -> None:
    track = Track(track_id=1, detection=make_detection())
    missed = track.mark_missed()

    assert missed.state == TrackState.LOST
    assert missed.age == 2
    assert missed.hits == 1
    assert missed.missed_frames == 1


def test_tracking_result_active_tracks() -> None:
    active_track = Track(
        track_id=1, detection=make_detection(), state=TrackState.ACTIVE)
    lost_track = Track(track_id=2, detection=make_detection(),
                       state=TrackState.LOST)

    result = TrackingResult(frame_index=1, timestamp=0.1,
                            tracks=(active_track, lost_track))

    assert result.active_tracks() == (active_track,)
