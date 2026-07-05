from basketball_vision.detection import BoundingBox, Detection, DetectionResult
from basketball_vision.tracking import SimpleTracker, TrackerConfig, TrackState


def make_detection_result(
    frame_index: int,
    detections: tuple[Detection, ...],
) -> DetectionResult:
    return DetectionResult(
        frame_index=frame_index,
        timestamp=float(frame_index) / 30.0,
        detections=detections,
    )


def make_detection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    class_id: int = 0,
) -> Detection:
    return Detection(
        bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=0.9,
        class_id=class_id,
        class_name="player",
    )


def test_simple_tracker_assigns_new_track_id() -> None:
    tracker = SimpleTracker()

    result = tracker.update(
        make_detection_result(
            frame_index=0,
            detections=(make_detection(10.0, 10.0, 50.0, 50.0),),
        )
    )

    assert len(result.tracks) == 1
    assert result.tracks[0].track_id == 1
    assert result.tracks[0].state == TrackState.NEW


def test_simple_tracker_preserves_track_id_on_iou_match() -> None:
    tracker = SimpleTracker(config=TrackerConfig(iou_threshold=0.3))

    first = tracker.update(
        make_detection_result(
            frame_index=0,
            detections=(make_detection(10.0, 10.0, 50.0, 50.0),),
        )
    )
    second = tracker.update(
        make_detection_result(
            frame_index=1,
            detections=(make_detection(12.0, 12.0, 52.0, 52.0),),
        )
    )

    assert first.tracks[0].track_id == second.tracks[0].track_id
    assert second.tracks[0].state == TrackState.ACTIVE
    assert second.tracks[0].hits == 2


def test_simple_tracker_creates_new_track_when_iou_is_too_low() -> None:
    tracker = SimpleTracker(config=TrackerConfig(iou_threshold=0.5))

    tracker.update(
        make_detection_result(
            frame_index=0,
            detections=(make_detection(10.0, 10.0, 50.0, 50.0),),
        )
    )
    result = tracker.update(
        make_detection_result(
            frame_index=1,
            detections=(make_detection(200.0, 200.0, 250.0, 250.0),),
        )
    )

    track_ids = {track.track_id for track in result.tracks}

    assert track_ids == {1, 2}


def test_simple_tracker_marks_unmatched_tracks_lost() -> None:
    tracker = SimpleTracker(config=TrackerConfig(max_missed_frames=2))

    tracker.update(
        make_detection_result(
            frame_index=0,
            detections=(make_detection(10.0, 10.0, 50.0, 50.0),),
        )
    )
    result = tracker.update(
        make_detection_result(frame_index=1, detections=()))

    assert len(result.tracks) == 1
    assert result.tracks[0].state == TrackState.LOST
    assert result.tracks[0].missed_frames == 1


def test_simple_tracker_removes_tracks_after_max_missed_frames() -> None:
    tracker = SimpleTracker(config=TrackerConfig(max_missed_frames=1))

    tracker.update(
        make_detection_result(
            frame_index=0,
            detections=(make_detection(10.0, 10.0, 50.0, 50.0),),
        )
    )
    tracker.update(make_detection_result(frame_index=1, detections=()))
    result = tracker.update(
        make_detection_result(frame_index=2, detections=()))

    assert result.tracks == ()


def test_simple_tracker_reset_clears_tracks_and_restarts_ids() -> None:
    tracker = SimpleTracker()

    tracker.update(
        make_detection_result(
            frame_index=0,
            detections=(make_detection(10.0, 10.0, 50.0, 50.0),),
        )
    )
    tracker.reset()
    result = tracker.update(
        make_detection_result(
            frame_index=1,
            detections=(make_detection(100.0, 100.0, 140.0, 140.0),),
        )
    )

    assert len(result.tracks) == 1
    assert result.tracks[0].track_id == 1
