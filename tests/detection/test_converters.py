"""Tests for YOLO conversion utilities."""

from basketball_vision.detection import YOLOConverter


class FakeTensor:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class FakeXYXY:
    def __init__(self, coords):
        self._coords = coords

    def tolist(self):
        return self._coords


class FakeBox:
    def __init__(
        self,
        coords,
        confidence,
        class_id,
    ):
        self.xyxy = [FakeXYXY(coords)]
        self.conf = FakeTensor(confidence)
        self.cls = FakeTensor(class_id)


class FakeResults:
    def __init__(self):
        self.names = {
            0: "person",
            1: "basketball",
        }

        self.boxes = [
            FakeBox(
                [10, 20, 110, 220],
                0.95,
                0,
            ),
            FakeBox(
                [250, 300, 280, 330],
                0.88,
                1,
            ),
        ]


def test_conversion():

    result = YOLOConverter.to_detection_result(
        results=FakeResults(),
        frame_index=5,
        timestamp=1.25,
    )

    assert result.frame_index == 5
    assert result.timestamp == 1.25
    assert result.num_detections == 2

    player = result.detections[0]

    assert player.class_name == "person"
    assert player.confidence == 0.95
    assert player.bounding_box.width == 100
