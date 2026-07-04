"""
Data model representing a single video frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """
    Immutable representation of a decoded video frame.

    Attributes:
        index:
            Zero-based frame index.

        timestamp:
            Timestamp in seconds.

        image:
            Image data in OpenCV BGR format.
    """

    index: int
    timestamp: float
    image: np.ndarray
