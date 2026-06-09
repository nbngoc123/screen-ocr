"""
CRNN Text Recognizer Module
"""

from .core import CRNNRecognizer
from .model import CRNN
from .postprocess import postprocess

__all__ = ["CRNNRecognizer", "CRNN", "postprocess"]
