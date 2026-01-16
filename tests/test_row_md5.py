"""
Tests for Row MD5 Scramble algorithm.

MD5 hash-based row pixel scrambling.
"""

import numpy as np

import pyscramble


class TestRowMD5:
    """Test suite for row_md5_encrypt and row_md5_decrypt."""

    def test_encrypt_decrypt_roundtrip(self, small_image):
        """Test that encrypt followed by decrypt returns original image."""
        height, width = small_image.shape[:2]
        key = "test_key"

        encrypted = pyscramble.row_md5_encrypt(small_image, width, height, key)
        decrypted = pyscramble.row_md5_decrypt(encrypted, width, height, key)

        np.testing.assert_array_equal(decrypted, small_image)

    def test_encrypt_changes_image(self, small_image):
        """Test that encryption actually changes the image."""
        height, width = small_image.shape[:2]
        key = "test_key"

        encrypted = pyscramble.row_md5_encrypt(small_image, width, height, key)

        assert not np.array_equal(encrypted, small_image)

    def test_different_keys_produce_different_results(self, small_image):
        """Test that different keys produce different encrypted images."""
        height, width = small_image.shape[:2]

        encrypted1 = pyscramble.row_md5_encrypt(small_image, width, height, "key1")
        encrypted2 = pyscramble.row_md5_encrypt(small_image, width, height, "key2")

        assert not np.array_equal(encrypted1, encrypted2)

    def test_same_key_produces_same_result(self, small_image):
        """Test that same key produces identical encrypted images."""
        height, width = small_image.shape[:2]
        key = "consistent_key"

        encrypted1 = pyscramble.row_md5_encrypt(small_image, width, height, key)
        encrypted2 = pyscramble.row_md5_encrypt(small_image, width, height, key)

        np.testing.assert_array_equal(encrypted1, encrypted2)

    def test_output_shape_preserved(self, small_image):
        """Test that output shape matches input shape."""
        height, width = small_image.shape[:2]
        key = "test_key"

        encrypted = pyscramble.row_md5_encrypt(small_image, width, height, key)

        assert encrypted.shape == small_image.shape

    def test_medium_image(self, medium_image):
        """Test with medium-sized image."""
        height, width = medium_image.shape[:2]
        key = "medium_test"

        encrypted = pyscramble.row_md5_encrypt(medium_image, width, height, key)
        decrypted = pyscramble.row_md5_decrypt(encrypted, width, height, key)

        np.testing.assert_array_equal(decrypted, medium_image)

    def test_non_square_image(self, non_square_image):
        """Test with non-square image dimensions."""
        height, width = non_square_image.shape[:2]
        key = "non_square_key"

        encrypted = pyscramble.row_md5_encrypt(non_square_image, width, height, key)
        decrypted = pyscramble.row_md5_decrypt(encrypted, width, height, key)

        np.testing.assert_array_equal(decrypted, non_square_image)

    def test_empty_key(self, small_image):
        """Test with empty string key."""
        height, width = small_image.shape[:2]
        key = ""

        encrypted = pyscramble.row_md5_encrypt(small_image, width, height, key)
        decrypted = pyscramble.row_md5_decrypt(encrypted, width, height, key)

        np.testing.assert_array_equal(decrypted, small_image)

    def test_unicode_key(self, small_image):
        """Test with unicode string key."""
        height, width = small_image.shape[:2]
        key = "行混淆测试🔐"

        encrypted = pyscramble.row_md5_encrypt(small_image, width, height, key)
        decrypted = pyscramble.row_md5_decrypt(encrypted, width, height, key)

        np.testing.assert_array_equal(decrypted, small_image)

    def test_wrong_key_fails_decrypt(self, small_image):
        """Test that wrong key does not correctly decrypt."""
        height, width = small_image.shape[:2]

        encrypted = pyscramble.row_md5_encrypt(small_image, width, height, "correct_key")
        decrypted = pyscramble.row_md5_decrypt(encrypted, width, height, "wrong_key")

        assert not np.array_equal(decrypted, small_image)

    def test_pixel_values_preserved(self, gradient_image):
        """Test that pixel values are preserved (just rearranged)."""
        height, width = gradient_image.shape[:2]
        key = "test_key"

        encrypted = pyscramble.row_md5_encrypt(gradient_image, width, height, key)

        # All pixel values should be preserved, just rearranged
        original_sorted = np.sort(gradient_image.flatten())
        encrypted_sorted = np.sort(encrypted.flatten())

        np.testing.assert_array_equal(original_sorted, encrypted_sorted)

    def test_row_scrambling_preserves_row_pixels(self, gradient_image):
        """Test that row scrambling only rearranges pixels within rows."""
        height, width = gradient_image.shape[:2]
        key = "row_test"

        encrypted = pyscramble.row_md5_encrypt(gradient_image, width, height, key)

        # For each row, sorted pixels should match
        for row in range(height):
            original_row_sorted = np.sort(gradient_image[row].flatten())
            encrypted_row_sorted = np.sort(encrypted[row].flatten())
            np.testing.assert_array_equal(original_row_sorted, encrypted_row_sorted)
