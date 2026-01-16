"""
Shared fixtures for pyscramble tests.
"""

import numpy as np
import pytest


@pytest.fixture
def small_image():
    """Create a small 8x8 RGBA test image with random pixels."""
    np.random.seed(42)
    return np.random.randint(0, 256, (8, 8, 4), dtype=np.uint8)


@pytest.fixture
def medium_image():
    """Create a medium 64x64 RGBA test image with random pixels."""
    np.random.seed(42)
    return np.random.randint(0, 256, (64, 64, 4), dtype=np.uint8)


@pytest.fixture
def large_image():
    """Create a large 256x256 RGBA test image with random pixels."""
    np.random.seed(42)
    return np.random.randint(0, 256, (256, 256, 4), dtype=np.uint8)


@pytest.fixture
def gradient_image():
    """Create a gradient RGBA test image for visual verification."""
    height, width = 32, 32
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            pixels[y, x] = [x * 8, y * 8, 128, 255]
    return pixels


@pytest.fixture
def non_square_image():
    """Create a non-square 32x64 RGBA test image."""
    np.random.seed(42)
    return np.random.randint(0, 256, (64, 32, 4), dtype=np.uint8)
