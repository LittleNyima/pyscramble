"""
pyscramble - High-performance image pixel scrambling algorithms

This module implements various image pixel scramble/descramble algorithms:
1. Tomato Scramble: Based on Gilbert 2D space-filling curve
2. Per Pixel MD5 Scramble: MD5 hash-based per-pixel scrambling
3. Row MD5 Scramble: MD5 hash-based row pixel scrambling
4. Block MD5 Scramble: MD5 hash-based block scrambling
5. Row Logistic Scramble: Logistic map-based row scrambling
6. Row Column Logistic Scramble: Logistic map-based row-column scrambling
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

import pyscramble._core as _core

__all__ = [
    # Tomato Scramble
    "tomato_scramble_encrypt",
    "tomato_scramble_decrypt",
    # Per Pixel MD5 Scramble
    "per_pixel_md5_encrypt",
    "per_pixel_md5_decrypt",
    # Row MD5 Scramble
    "row_md5_encrypt",
    "row_md5_decrypt",
    # Block MD5 Scramble
    "block_md5_encrypt",
    "block_md5_decrypt",
    # Row Logistic Scramble
    "row_logistic_encrypt",
    "row_logistic_decrypt",
    # Row Column Logistic Scramble
    "row_column_logistic_encrypt",
    "row_column_logistic_decrypt",
]


# ==================== Internal Helper Functions ====================


def _pixels_to_int(pixels: NDArray[np.uint8]) -> NDArray[np.int32]:
    """Convert RGBA pixel array to integer array."""
    pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
    return _core.pixels_to_int(pixels)


def _int_to_pixels(int_pixels: NDArray[np.int32], width: int, height: int) -> NDArray[np.uint8]:
    """Convert integer array back to RGBA pixel array."""
    flat = _core.int_to_pixels(int_pixels)
    return flat.reshape(height, width, -1)


# ==================== Tomato Scramble ====================


def tomato_scramble_encrypt(
    pixels: NDArray[np.uint8],
    width: int,
    height: int,
    key: float = 1.0,
) -> NDArray[np.uint8]:
    """
    Encrypt pixels using Tomato Scramble algorithm.

    Based on Gilbert 2D space-filling curve.

    Args:
        pixels: Input pixel array with shape (height, width, 4)
        width: Image width
        height: Image height
        key: Encryption key (default 1.0)

    Returns:
        Encrypted pixel array
    """
    int_pixels = _pixels_to_int(pixels)
    result = _core.tomato_scramble_encrypt(int_pixels, width, height, key)
    return _int_to_pixels(result, width, height)


def tomato_scramble_decrypt(
    pixels: NDArray[np.uint8],
    width: int,
    height: int,
    key: float = 1.0,
) -> NDArray[np.uint8]:
    """
    Decrypt pixels using Tomato Scramble algorithm.

    Based on Gilbert 2D space-filling curve.

    Args:
        pixels: Encrypted pixel array with shape (height, width, 4)
        width: Image width
        height: Image height
        key: Decryption key (default 1.0)

    Returns:
        Decrypted pixel array
    """
    int_pixels = _pixels_to_int(pixels)
    result = _core.tomato_scramble_decrypt(int_pixels, width, height, key)
    return _int_to_pixels(result, width, height)


# ==================== Per Pixel MD5 Scramble ====================


def per_pixel_md5_encrypt(
    pixels: NDArray[np.uint8],
    width: int,
    height: int,
    key: str,
) -> NDArray[np.uint8]:
    """
    Encrypt pixels using Per-Pixel MD5 Scramble algorithm.

    MD5 hash-based per-pixel scrambling.

    Args:
        pixels: Input pixel array with shape (height, width, 4)
        width: Image width
        height: Image height
        key: Encryption key (string)

    Returns:
        Encrypted pixel array
    """
    int_pixels = _pixels_to_int(pixels)
    result = _core.per_pixel_md5_encrypt(int_pixels, width, height, key)
    return _int_to_pixels(result, width, height)


def per_pixel_md5_decrypt(
    pixels: NDArray[np.uint8],
    width: int,
    height: int,
    key: str,
) -> NDArray[np.uint8]:
    """
    Decrypt pixels using Per-Pixel MD5 Scramble algorithm.

    MD5 hash-based per-pixel scrambling.

    Args:
        pixels: Encrypted pixel array with shape (height, width, 4)
        width: Image width
        height: Image height
        key: Decryption key (string)

    Returns:
        Decrypted pixel array
    """
    int_pixels = _pixels_to_int(pixels)
    result = _core.per_pixel_md5_decrypt(int_pixels, width, height, key)
    return _int_to_pixels(result, width, height)


# ==================== Row MD5 Scramble ====================


def row_md5_encrypt(
    pixels: NDArray[np.uint8],
    width: int,
    height: int,
    key: str,
) -> NDArray[np.uint8]:
    """
    Encrypt pixels using Row MD5 Scramble algorithm.

    MD5 hash-based row pixel scrambling.

    Args:
        pixels: Input pixel array with shape (height, width, 4)
        width: Image width
        height: Image height
        key: Encryption key (string)

    Returns:
        Encrypted pixel array
    """
    int_pixels = _pixels_to_int(pixels)
    result = _core.row_md5_encrypt(int_pixels, width, height, key)
    return _int_to_pixels(result, width, height)


def row_md5_decrypt(
    pixels: NDArray[np.uint8],
    width: int,
    height: int,
    key: str,
) -> NDArray[np.uint8]:
    """
    Decrypt pixels using Row MD5 Scramble algorithm.

    MD5 hash-based row pixel scrambling.

    Args:
        pixels: Encrypted pixel array with shape (height, width, 4)
        width: Image width
        height: Image height
        key: Decryption key (string)

    Returns:
        Decrypted pixel array
    """
    int_pixels = _pixels_to_int(pixels)
    result = _core.row_md5_decrypt(int_pixels, width, height, key)
    return _int_to_pixels(result, width, height)


# ==================== Block MD5 Scramble ====================


def block_md5_encrypt(
    pixels: NDArray[np.uint8],
    width: int,
    height: int,
    key: str,
    x_block_count: int = 32,
    y_block_count: int = 32,
) -> Tuple[NDArray[np.uint8], int, int]:
    """
    Encrypt pixels using Block MD5 Scramble algorithm.

    MD5 hash-based block scrambling. Image may be padded to fit block size.

    Args:
        pixels: Input pixel array with shape (height, width, 4)
        width: Image width
        height: Image height
        key: Encryption key (string)
        x_block_count: Number of blocks in x direction (default 32)
        y_block_count: Number of blocks in y direction (default 32)

    Returns:
        Tuple of (encrypted_pixels, new_width, new_height)
        New dimensions may differ from original due to padding.
    """
    int_pixels = _pixels_to_int(pixels)
    new_pixels, new_width, new_height = _core.block_md5_encrypt(
        int_pixels,
        width,
        height,
        key,
        x_block_count,
        y_block_count,
    )
    return _int_to_pixels(new_pixels, new_width, new_height), new_width, new_height


def block_md5_decrypt(
    pixels: NDArray[np.uint8],
    width: int,
    height: int,
    key: str,
    x_block_count: int = 32,
    y_block_count: int = 32,
) -> Tuple[NDArray[np.uint8], int, int]:
    """
    Decrypt pixels using Block MD5 Scramble algorithm.

    MD5 hash-based block scrambling.

    Args:
        pixels: Encrypted pixel array with shape (height, width, 4)
        width: Image width (may be padded width from encryption)
        height: Image height (may be padded height from encryption)
        key: Decryption key (string)
        x_block_count: Number of blocks in x direction (default 32)
        y_block_count: Number of blocks in y direction (default 32)

    Returns:
        Tuple of (decrypted_pixels, output_width, output_height)
    """
    int_pixels = _pixels_to_int(pixels)
    new_pixels = _core.block_md5_decrypt(int_pixels, width, height, key, x_block_count, y_block_count)
    result = _int_to_pixels(new_pixels, width, height)

    return result, width, height


# ==================== Row Logistic Scramble ====================


def row_logistic_encrypt(
    pixels: NDArray[np.uint8],
    width: int,
    height: int,
    key: float,
) -> NDArray[np.uint8]:
    """
    Encrypt pixels using Row Logistic Scramble algorithm.

    Logistic map-based row scrambling.

    Args:
        pixels: Input pixel array with shape (height, width, 4)
        width: Image width
        height: Image height
        key: Encryption key (float, should be in range (0, 1))

    Returns:
        Encrypted pixel array
    """
    int_pixels = _pixels_to_int(pixels)
    result = _core.row_logistic_encrypt(int_pixels, width, height, key)
    return _int_to_pixels(result, width, height)


def row_logistic_decrypt(
    pixels: NDArray[np.uint8],
    width: int,
    height: int,
    key: float,
) -> NDArray[np.uint8]:
    """
    Decrypt pixels using Row Logistic Scramble algorithm.

    Logistic map-based row scrambling.

    Args:
        pixels: Encrypted pixel array with shape (height, width, 4)
        width: Image width
        height: Image height
        key: Decryption key (float, should be in range (0, 1))

    Returns:
        Decrypted pixel array
    """
    int_pixels = _pixels_to_int(pixels)
    result = _core.row_logistic_decrypt(int_pixels, width, height, key)
    return _int_to_pixels(result, width, height)


# ==================== Row Column Logistic Scramble ====================


def row_column_logistic_encrypt(
    pixels: NDArray[np.uint8],
    width: int,
    height: int,
    key: float,
) -> NDArray[np.uint8]:
    """
    Encrypt pixels using Row-Column Logistic Scramble algorithm.

    Logistic map-based row and column scrambling.

    Args:
        pixels: Input pixel array with shape (height, width, 4)
        width: Image width
        height: Image height
        key: Encryption key (float, should be in range (0, 1))

    Returns:
        Encrypted pixel array
    """
    int_pixels = _pixels_to_int(pixels)
    result = _core.row_column_logistic_encrypt(int_pixels, width, height, key)
    return _int_to_pixels(result, width, height)


def row_column_logistic_decrypt(
    pixels: NDArray[np.uint8],
    width: int,
    height: int,
    key: float,
) -> NDArray[np.uint8]:
    """
    Decrypt pixels using Row-Column Logistic Scramble algorithm.

    Logistic map-based row and column scrambling.

    Args:
        pixels: Encrypted pixel array with shape (height, width, 4)
        width: Image width
        height: Image height
        key: Decryption key (float, should be in range (0, 1))

    Returns:
        Decrypted pixel array
    """
    int_pixels = _pixels_to_int(pixels)
    result = _core.row_column_logistic_decrypt(int_pixels, width, height, key)
    return _int_to_pixels(result, width, height)
