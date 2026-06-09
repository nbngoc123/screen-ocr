from dataclasses import dataclass
from typing import Protocol
import numpy as np

@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0


class TextDetector(Protocol):
    """Protocol bắt buộc cho mọi text detector."""
    def detect(self, image: np.ndarray) -> list[BoundingBox]:
        ...
