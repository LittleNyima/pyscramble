"""
Tests for Per-Pixel MD5 Scramble algorithm.

MD5 hash-based per-pixel scrambling.
"""

import numpy as np

import pyscramble


class TestPerPixelMD5:
    """Test suite for per_pixel_md5_encrypt and per_pixel_md5_decrypt."""

    def test_encrypt_decrypt_roundtrip(self, small_image):
        """Test that encrypt followed by decrypt returns original image."""
        key = "test_key"

        encrypted = pyscramble.per_pixel_md5_encrypt(small_image, key)
        decrypted = pyscramble.per_pixel_md5_decrypt(encrypted, key)

        np.testing.assert_array_equal(decrypted, small_image)

    def test_encrypt_changes_image(self, small_image):
        """Test that encryption actually changes the image."""
        key = "test_key"

        encrypted = pyscramble.per_pixel_md5_encrypt(small_image, key)

        assert not np.array_equal(encrypted, small_image)

    def test_different_keys_produce_different_results(self, small_image):
        """Test that different keys produce different encrypted images."""
        encrypted1 = pyscramble.per_pixel_md5_encrypt(small_image, "key1")
        encrypted2 = pyscramble.per_pixel_md5_encrypt(small_image, "key2")

        assert not np.array_equal(encrypted1, encrypted2)

    def test_same_key_produces_same_result(self, small_image):
        """Test that same key produces identical encrypted images."""
        key = "consistent_key"

        encrypted1 = pyscramble.per_pixel_md5_encrypt(small_image, key)
        encrypted2 = pyscramble.per_pixel_md5_encrypt(small_image, key)

        np.testing.assert_array_equal(encrypted1, encrypted2)

    def test_output_shape_preserved(self, small_image):
        """Test that output shape matches input shape."""
        key = "test_key"

        encrypted = pyscramble.per_pixel_md5_encrypt(small_image, key)

        assert encrypted.shape == small_image.shape

    def test_medium_image(self, medium_image):
        """Test with medium-sized image."""
        key = "medium_test"

        encrypted = pyscramble.per_pixel_md5_encrypt(medium_image, key)
        decrypted = pyscramble.per_pixel_md5_decrypt(encrypted, key)

        np.testing.assert_array_equal(decrypted, medium_image)

    def test_non_square_image(self, non_square_image):
        """Test with non-square image dimensions."""
        key = "non_square_key"

        encrypted = pyscramble.per_pixel_md5_encrypt(non_square_image, key)
        decrypted = pyscramble.per_pixel_md5_decrypt(encrypted, key)

        np.testing.assert_array_equal(decrypted, non_square_image)

    def test_empty_key(self, small_image):
        """Test with empty string key."""
        key = ""

        encrypted = pyscramble.per_pixel_md5_encrypt(small_image, key)
        decrypted = pyscramble.per_pixel_md5_decrypt(encrypted, key)

        np.testing.assert_array_equal(decrypted, small_image)

    def test_unicode_key(self, small_image):
        """Test with unicode string key."""
        key = "密钥测试🔑"

        encrypted = pyscramble.per_pixel_md5_encrypt(small_image, key)
        decrypted = pyscramble.per_pixel_md5_decrypt(encrypted, key)

        np.testing.assert_array_equal(decrypted, small_image)

    def test_wrong_key_fails_decrypt(self, small_image):
        """Test that wrong key does not correctly decrypt."""
        encrypted = pyscramble.per_pixel_md5_encrypt(small_image, "correct_key")
        decrypted = pyscramble.per_pixel_md5_decrypt(encrypted, "wrong_key")

        assert not np.array_equal(decrypted, small_image)

    def test_pixel_values_preserved(self, gradient_image):
        """Test that pixel values are preserved (just rearranged)."""
        key = "test_key"

        encrypted = pyscramble.per_pixel_md5_encrypt(gradient_image, key)

        # All pixel values should be preserved, just rearranged
        original_sorted = np.sort(gradient_image.flatten())
        encrypted_sorted = np.sort(encrypted.flatten())

        np.testing.assert_array_equal(original_sorted, encrypted_sorted)

    def test_long_key(self, small_image):
        """Test with very long key string."""
        key = "a" * 10000

        encrypted = pyscramble.per_pixel_md5_encrypt(small_image, key)
        decrypted = pyscramble.per_pixel_md5_decrypt(encrypted, key)

        np.testing.assert_array_equal(decrypted, small_image)
