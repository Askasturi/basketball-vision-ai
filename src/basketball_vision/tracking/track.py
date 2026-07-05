from dataclasses import dataclass

from basketball_vision.detection import Detection
from basketball_vision.tracking.types import TrackState


@dataclass(frozen=True, slots=True)
class Track:
    track_id: int
    detection: Detection
    state: TrackState = TrackState.NEW
    age: int = 1
    hits: int = 1
    missed_frames: int = 0

    def mark_updated(self, detection: Detection) -> "Track":
        return Track(
            track_id=self.track_id,
            detection=detection,
            state=TrackState.ACTIVE,
            age=self.age + 1,
            hits=self.hits + 1,
            missed_frames=0,
        )

    def mark_missed(self) -> "Track":
        return Track(
            track_id=self.track_id,
            detection=self.detection,
            state=TrackState.LOST,
            age=self.age + 1,
            hits=self.hits,
            missed_frames=self.missed_frames + 1,
        )

    def mark_removed(self) -> "Track":
        return Track(
            track_id=self.track_id,
            detection=self.detection,
            state=TrackState.REMOVED,
            age=self.age,
            hits=self.hits,
            missed_frames=self.missed_frames,
        )


@dataclass(frozen=True, slots=True)
class TrackingResult:
    frame_index: int
    timestamp: float | None
    tracks: tuple[Track, ...]

    def active_tracks(self) -> tuple[Track, ...]:
        return tuple(
            track
            for track in self.tracks
            if track.state in {TrackState.NEW, TrackState.ACTIVE}
        )
