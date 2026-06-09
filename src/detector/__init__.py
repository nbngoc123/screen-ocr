"""
Detector module for Text Detection.
Provides DBNetDetector and BoundingBox structures.
"""

from .types import BoundingBox, TextDetector
from .core import DBNetDetector

__all__ = ["BoundingBox", "TextDetector", "DBNetDetector"]
