"""Configuration for basketball analytics."""

from dataclasses import dataclass
from pathlib import Path

from basketball_vision.analytics.exceptions import AnalyticsConfigurationError


@dataclass(frozen=True, slots=True)
class AnalyticsConfig:
    """Configuration for analytics processing and export."""

    fps: float = 30.0
    output_dir: Path = Path("outputs/analytics")
    export_csv: bool = True
    export_json: bool = True

    def __post_init__(self) -> None:
        """Validate analytics configuration."""
        if self.fps <= 0:
            msg = "fps must be greater than 0."
            raise AnalyticsConfigurationError(msg)

        if not isinstance(self.output_dir, Path):
            object.__setattr__(self, "output_dir", Path(self.output_dir))
