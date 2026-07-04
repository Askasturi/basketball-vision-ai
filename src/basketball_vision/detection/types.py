"""Shared types for the detection framework."""

from enum import StrEnum


class DetectorType(StrEnum):
    """Supported detector backends."""

    YOLO = "yolo"
    RTDETR = "rtdetr"
    ONNX = "onnx"
    TENSORRT = "tensorrt"


class DeviceType(StrEnum):
    """Supported inference devices."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
