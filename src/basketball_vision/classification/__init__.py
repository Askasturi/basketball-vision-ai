"""Player team classification package."""

from basketball_vision.classification.base_classifier import BasePlayerClassifier
from basketball_vision.classification.color_team_classifier import ColorTeamClassifier
from basketball_vision.classification.config import (
    ClassificationConfig,
    ColorTeamClassifierConfig,
)
from basketball_vision.classification.exceptions import (
    ClassificationError,
    ClassifierNotRegisteredError,
    InvalidClassificationConfigError,
)
from basketball_vision.classification.factory import PlayerClassifierFactory
from basketball_vision.classification.player_team import (
    ClassificationResult,
    PlayerTeamAssignment,
)
from basketball_vision.classification.types import ClassifierType, TeamLabel

__all__ = [
    "BasePlayerClassifier",
    "ClassificationConfig",
    "ClassificationError",
    "ClassificationResult",
    "ClassifierNotRegisteredError",
    "ClassifierType",
    "ColorTeamClassifier",
    "ColorTeamClassifierConfig",
    "InvalidClassificationConfigError",
    "PlayerClassifierFactory",
    "PlayerTeamAssignment",
    "TeamLabel",
]
