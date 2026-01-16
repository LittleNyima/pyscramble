"""
pyscramble._core - High-performance image pixel scrambling algorithms

Rust implementation of various image pixel scramble/descramble algorithms.
"""

import numpy as np
from numpy.typing import NDArray

def pixels_to_int(pixels: NDArray[np.uint8]) -> NDArray[np.int32]:
    """
    Convert RGBA pixel array to integer array.

    Args:
        pixels: RGBA pixel array with shape (height, width, 4)

    Returns:
        Flattened integer array with shape (height * width,)
        Format: A<<24 | R<<16 | G<<8 | B
    """
    ...

def int_to_pixels(int_pixels: NDArray[np.int32]) -> NDArray[np.uint8]:
    """
    Convert integer array back to RGBA pixel array.

    Args:
        int_pixels: Integer array with shape (height * width,)

    Returns:
        Flattened RGBA pixel array with shape (height * width * 4,)
        Python side needs to reshape to (height, width, 4)
    """
    ...

def tomato_scramble_encrypt(
    int_pixels: NDArray[np.int32],
    width: int,
    height: int,
    key: float,
) -> NDArray[np.int32]:
    """
    Encrypt pixels using Tomato Scramble algorithm.

    Based on Gilbert 2D space-filling curve.

    Args:
        int_pixels: Integer pixel array with shape (height * width,)
        width: Image width
        height: Image height
        key: Encryption key (float)

    Returns:
        Encrypted integer pixel array
    """
    ...

def tomato_scramble_decrypt(
    int_pixels: NDArray[np.int32],
    width: int,
    height: int,
    key: float,
) -> NDArray[np.int32]:
    """
    Decrypt pixels using Tomato Scramble algorithm.

    Based on Gilbert 2D space-filling curve.

    Args:
        int_pixels: Encrypted integer pixel array with shape (height * width,)
        width: Image width
        height: Image height
        key: Decryption key (float)

    Returns:
        Decrypted integer pixel array
    """
    ...

def per_pixel_md5_encrypt(
    int_pixels: NDArray[np.int32],
    width: int,
    height: int,
    key: str,
) -> NDArray[np.int32]:
    """
    Encrypt pixels using Per-Pixel MD5 Scramble algorithm.

    MD5 hash-based per-pixel scrambling.

    Args:
        int_pixels: Integer pixel array with shape (height * width,)
        width: Image width
        height: Image height
        key: Encryption key (string)

    Returns:
        Encrypted integer pixel array
    """
    ...

def per_pixel_md5_decrypt(
    int_pixels: NDArray[np.int32],
    width: int,
    height: int,
    key: str,
) -> NDArray[np.int32]:
    """
    Decrypt pixels using Per-Pixel MD5 Scramble algorithm.

    MD5 hash-based per-pixel scrambling.

    Args:
        int_pixels: Encrypted integer pixel array with shape (height * width,)
        width: Image width
        height: Image height
        key: Decryption key (string)

    Returns:
        Decrypted integer pixel array
    """
    ...

def row_md5_encrypt(
    int_pixels: NDArray[np.int32],
    width: int,
    height: int,
    key: str,
) -> NDArray[np.int32]:
    """
    Encrypt pixels using Row MD5 Scramble algorithm.

    MD5 hash-based row pixel scrambling.

    Args:
        int_pixels: Integer pixel array with shape (height * width,)
        width: Image width
        height: Image height
        key: Encryption key (string)

    Returns:
        Encrypted integer pixel array
    """
    ...

def row_md5_decrypt(
    int_pixels: NDArray[np.int32],
    width: int,
    height: int,
    key: str,
) -> NDArray[np.int32]:
    """
    Decrypt pixels using Row MD5 Scramble algorithm.

    MD5 hash-based row pixel scrambling.

    Args:
        int_pixels: Encrypted integer pixel array with shape (height * width,)
        width: Image width
        height: Image height
        key: Decryption key (string)

    Returns:
        Decrypted integer pixel array
    """
    ...

def block_md5_encrypt(
    int_pixels: NDArray[np.int32],
    width: int,
    height: int,
    key: str,
    x_block_count: int,
    y_block_count: int,
) -> tuple[NDArray[np.int32], int, int]:
    """
    Encrypt pixels using Block MD5 Scramble algorithm.

    MD5 hash-based block scrambling. Image may be padded to fit block size.

    Args:
        int_pixels: Integer pixel array with shape (height * width,)
        width: Image width
        height: Image height
        key: Encryption key (string)
        x_block_count: Number of blocks in x direction
        y_block_count: Number of blocks in y direction

    Returns:
        Tuple of (encrypted_pixels, new_width, new_height)
        New dimensions may differ from original due to padding.
    """
    ...

def block_md5_decrypt(
    int_pixels: NDArray[np.int32],
    width: int,
    height: int,
    key: str,
    x_block_count: int,
    y_block_count: int,
) -> NDArray[np.int32]:
    """
    Decrypt pixels using Block MD5 Scramble algorithm.

    MD5 hash-based block scrambling.

    Args:
        int_pixels: Encrypted integer pixel array with shape (height * width,)
        width: Image width (may be padded width from encryption)
        height: Image height (may be padded height from encryption)
        key: Decryption key (string)
        x_block_count: Number of blocks in x direction
        y_block_count: Number of blocks in y direction

    Returns:
        Decrypted integer pixel array
    """
    ...

def row_logistic_encrypt(
    int_pixels: NDArray[np.int32],
    width: int,
    height: int,
    key: float,
) -> NDArray[np.int32]:
    """
    Encrypt pixels using Row Logistic Scramble algorithm.

    Logistic map-based row scrambling.

    Args:
        int_pixels: Integer pixel array with shape (height * width,)
        width: Image width
        height: Image height
        key: Encryption key (float, should be in range (0, 1))

    Returns:
        Encrypted integer pixel array
    """
    ...

def row_logistic_decrypt(
    int_pixels: NDArray[np.int32],
    width: int,
    height: int,
    key: float,
) -> NDArray[np.int32]:
    """
    Decrypt pixels using Row Logistic Scramble algorithm.

    Logistic map-based row scrambling.

    Args:
        int_pixels: Encrypted integer pixel array with shape (height * width,)
        width: Image width
        height: Image height
        key: Decryption key (float, should be in range (0, 1))

    Returns:
        Decrypted integer pixel array
    """
    ...

def row_column_logistic_encrypt(
    int_pixels: NDArray[np.int32],
    width: int,
    height: int,
    key: float,
) -> NDArray[np.int32]:
    """
    Encrypt pixels using Row-Column Logistic Scramble algorithm.

    Logistic map-based row and column scrambling.

    Args:
        int_pixels: Integer pixel array with shape (height * width,)
        width: Image width
        height: Image height
        key: Encryption key (float, should be in range (0, 1))

    Returns:
        Encrypted integer pixel array
    """
    ...

def row_column_logistic_decrypt(
    int_pixels: NDArray[np.int32],
    width: int,
    height: int,
    key: float,
) -> NDArray[np.int32]:
    """
    Decrypt pixels using Row-Column Logistic Scramble algorithm.

    Logistic map-based row and column scrambling.

    Args:
        int_pixels: Encrypted integer pixel array with shape (height * width,)
        width: Image width
        height: Image height
        key: Decryption key (float, should be in range (0, 1))

    Returns:
        Decrypted integer pixel array
    """
    ...
