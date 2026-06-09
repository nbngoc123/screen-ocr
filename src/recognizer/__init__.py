"""
CRNN Text Recognizer Module
"""

from .core import CRNNRecognizer
from .model import CRNN
from .postprocess import CTCDecoder

__all__ = ["CRNNRecognizer", "CRNN", "CTCDecoder"]
