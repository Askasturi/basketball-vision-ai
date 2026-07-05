"""Result objects for the unified basketball vision pipeline."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class PipelineFrameResult:
    """Result produced after processing one frame."""

    frame_index: int
    timestamp: float | None
    frame: np.ndarray
    detections: Any | None = None
    tracking: Any | None = None
    classification: Any | None = None
    recognition: Any | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            msg = "frame_index must be greater than or equal to 0."
            raise ValueError(msg)

        if self.timestamp is not None and self.timestamp < 0:
            msg = "timestamp must be greater than or equal to 0 when provided."
            raise ValueError(msg)

        if not isinstance(self.frame, np.ndarray):
            msg = "frame must be a numpy.ndarray."
            raise TypeError(msg)
