"""Factory for player classifiers."""

from collections.abc import Callable

from basketball_vision.classification.base_classifier import BasePlayerClassifier
from basketball_vision.classification.color_team_classifier import ColorTeamClassifier
from basketball_vision.classification.config import ClassificationConfig
from basketball_vision.classification.exceptions import ClassifierNotRegisteredError
from basketball_vision.classification.types import ClassifierType

ClassifierBuilder = Callable[[
    ClassificationConfig | None], BasePlayerClassifier]


class PlayerClassifierFactory:
    """Registry-backed factory for player classifiers."""

    _registry: dict[ClassifierType, type[BasePlayerClassifier]] = {
        ClassifierType.COLOR: ColorTeamClassifier,
    }

    @classmethod
    def register(
        cls,
        classifier_type: ClassifierType,
        classifier_class: type[BasePlayerClassifier],
    ) -> None:
        """Register a classifier implementation."""
        cls._registry[classifier_type] = classifier_class

    @classmethod
    def create(
        cls,
        classifier_type: ClassifierType | str,
        config: ClassificationConfig | None = None,
    ) -> BasePlayerClassifier:
        """Create a classifier by type."""
        normalized_type = ClassifierType(classifier_type)

        classifier_class = cls._registry.get(normalized_type)
        if classifier_class is None:
            msg = f"Classifier type is not registered: {normalized_type}"
            raise ClassifierNotRegisteredError(msg)

        return classifier_class(config)

    @classmethod
    def available_classifiers(cls) -> tuple[ClassifierType, ...]:
        """Return registered classifier types."""
        return tuple(cls._registry)
