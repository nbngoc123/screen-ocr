import pytest
from src.postprocess import postprocess

def test_postprocess_whitespace():
    assert postprocess("  hello  world  ") == "hello world"

def test_postprocess_zero_oh():
    assert postprocess("HELL0") == "HELLO"
