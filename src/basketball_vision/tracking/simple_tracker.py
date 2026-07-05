from basketball_vision.detection import BoundingBox, Detection, DetectionResult
from basketball_vision.tracking.base_tracker import BaseTracker
from basketball_vision.tracking.config import TrackerConfig
from basketball_vision.tracking.track import Track, TrackingResult
from basketball_vision.tracking.types import TrackState


class SimpleTracker(BaseTracker):
    """Simple IoU-based multi-object tracker.

    This baseline tracker:
    - greedily matches detections to existing tracks by IoU
    - assigns persistent integer track IDs
    - marks unmatched tracks as lost
    - removes tracks after too many missed frames
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        super().__init__(config)
        self._tracks: dict[int, Track] = {}
        self._next_track_id = 1

    def update(self, detection_result: DetectionResult) -> TrackingResult:
        detections = tuple(detection_result.detections)

        matched_track_ids: set[int] = set()
        matched_detection_indices: set[int] = set()
        updated_tracks: dict[int, Track] = {}

        matches = self._match_detections(detections)

        for track_id, detection_index in matches:
            track = self._tracks[track_id]
            detection = detections[detection_index]

            updated_tracks[track_id] = track.mark_updated(detection)
            matched_track_ids.add(track_id)
            matched_detection_indices.add(detection_index)

        for track_id, track in self._tracks.items():
            if track_id in matched_track_ids:
                continue

            missed_track = track.mark_missed()

            if missed_track.missed_frames > self.config.max_missed_frames:
                continue

            updated_tracks[track_id] = missed_track

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indices:
                continue

            new_track = Track(
                track_id=self._next_track_id,
                detection=detection,
                state=TrackState.NEW,
            )

            updated_tracks[new_track.track_id] = new_track
            self._next_track_id += 1

        self._tracks = updated_tracks

        return TrackingResult(
            frame_index=detection_result.frame_index,
            timestamp=detection_result.timestamp,
            tracks=tuple(
                sorted(
                    self._tracks.values(),
                    key=lambda track: track.track_id,
                )
            ),
        )

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    def _match_detections(
        self,
        detections: tuple[Detection, ...],
    ) -> list[tuple[int, int]]:
        candidates: list[tuple[float, int, int]] = []

        for track_id, track in self._tracks.items():
            if track.state == TrackState.REMOVED:
                continue

            for detection_index, detection in enumerate(detections):
                score = _bbox_iou(
                    track.detection.bounding_box,
                    detection.bounding_box,
                )

                if score >= self.config.iou_threshold:
                    candidates.append((score, track_id, detection_index))

        candidates.sort(
            reverse=True,
            key=lambda candidate: candidate[0],
        )

        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        matches: list[tuple[int, int]] = []

        for _, track_id, detection_index in candidates:
            if track_id in used_tracks:
                continue

            if detection_index in used_detections:
                continue

            matches.append((track_id, detection_index))
            used_tracks.add(track_id)
            used_detections.add(detection_index)

        return matches


def _bbox_iou(first: BoundingBox, second: BoundingBox) -> float:
    x_left = max(first.x1, second.x1)
    y_top = max(first.y1, second.y1)
    x_right = min(first.x2, second.x2)
    y_bottom = min(first.y2, second.y2)

    intersection_width = max(0.0, x_right - x_left)
    intersection_height = max(0.0, y_bottom - y_top)
    intersection_area = intersection_width * intersection_height

    first_area = max(0.0, first.x2 - first.x1) * max(
        0.0,
        first.y2 - first.y1,
    )
    second_area = max(0.0, second.x2 - second.x1) * max(
        0.0,
        second.y2 - second.y1,
    )

    union_area = first_area + second_area - intersection_area

    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area
