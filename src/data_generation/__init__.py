"""
Data Generation Module
"""

from .generator import generate_dataset, generate_sample
from .corpus import random_text
from .augment import preprocess_for_crnn

__all__ = ["generate_dataset", "generate_sample", "random_text", "preprocess_for_crnn"]
