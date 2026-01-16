"""
Tests for Row-Column Logistic Scramble algorithm.

Logistic map-based row and column scrambling.
"""

import numpy as np

import pyscramble


class TestRowColumnLogistic:
    """Test suite for row_column_logistic_encrypt and row_column_logistic_decrypt."""

    def test_encrypt_decrypt_roundtrip(self, small_image):
        """Test that encrypt followed by decrypt returns original image."""
        height, width = small_image.shape[:2]
        key = 0.5

        encrypted = pyscramble.row_column_logistic_encrypt(small_image, width, height, key)
        decrypted = pyscramble.row_column_logistic_decrypt(encrypted, width, height, key)

        np.testing.assert_array_equal(decrypted, small_image)

    def test_encrypt_changes_image(self, small_image):
        """Test that encryption actually changes the image."""
        height, width = small_image.shape[:2]
        key = 0.5

        encrypted = pyscramble.row_column_logistic_encrypt(small_image, width, height, key)

        assert not np.array_equal(encrypted, small_image)

    def test_different_keys_produce_different_results(self, small_image):
        """Test that different keys produce different encrypted images."""
        height, width = small_image.shape[:2]

        encrypted1 = pyscramble.row_column_logistic_encrypt(small_image, width, height, 0.3)
        encrypted2 = pyscramble.row_column_logistic_encrypt(small_image, width, height, 0.7)

        assert not np.array_equal(encrypted1, encrypted2)

    def test_same_key_produces_same_result(self, small_image):
        """Test that same key produces identical encrypted images."""
        height, width = small_image.shape[:2]
        key = 0.456

        encrypted1 = pyscramble.row_column_logistic_encrypt(small_image, width, height, key)
        encrypted2 = pyscramble.row_column_logistic_encrypt(small_image, width, height, key)

        np.testing.assert_array_equal(encrypted1, encrypted2)

    def test_output_shape_preserved(self, small_image):
        """Test that output shape matches input shape."""
        height, width = small_image.shape[:2]
        key = 0.5

        encrypted = pyscramble.row_column_logistic_encrypt(small_image, width, height, key)

        assert encrypted.shape == small_image.shape

    def test_medium_image(self, medium_image):
        """Test with medium-sized image."""
        height, width = medium_image.shape[:2]
        key = 0.123

        encrypted = pyscramble.row_column_logistic_encrypt(medium_image, width, height, key)
        decrypted = pyscramble.row_column_logistic_decrypt(encrypted, width, height, key)

        np.testing.assert_array_equal(decrypted, medium_image)

    def test_non_square_image(self, non_square_image):
        """Test with non-square image dimensions."""
        height, width = non_square_image.shape[:2]
        key = 0.789

        encrypted = pyscramble.row_column_logistic_encrypt(non_square_image, width, height, key)
        decrypted = pyscramble.row_column_logistic_decrypt(encrypted, width, height, key)

        np.testing.assert_array_equal(decrypted, non_square_image)

    def test_key_near_zero(self, small_image):
        """Test with key value close to 0."""
        height, width = small_image.shape[:2]
        key = 0.001

        encrypted = pyscramble.row_column_logistic_encrypt(small_image, width, height, key)
        decrypted = pyscramble.row_column_logistic_decrypt(encrypted, width, height, key)

        np.testing.assert_array_equal(decrypted, small_image)

    def test_key_near_one(self, small_image):
        """Test with key value close to 1."""
        height, width = small_image.shape[:2]
        key = 0.999

        encrypted = pyscramble.row_column_logistic_encrypt(small_image, width, height, key)
        decrypted = pyscramble.row_column_logistic_decrypt(encrypted, width, height, key)

        np.testing.assert_array_equal(decrypted, small_image)

    def test_wrong_key_fails_decrypt(self, small_image):
        """Test that wrong key does not correctly decrypt."""
        height, width = small_image.shape[:2]

        encrypted = pyscramble.row_column_logistic_encrypt(small_image, width, height, 0.5)
        decrypted = pyscramble.row_column_logistic_decrypt(encrypted, width, height, 0.6)

        assert not np.array_equal(decrypted, small_image)

    def test_pixel_values_preserved(self, gradient_image):
        """Test that pixel values are preserved (just rearranged)."""
        height, width = gradient_image.shape[:2]
        key = 0.5

        encrypted = pyscramble.row_column_logistic_encrypt(gradient_image, width, height, key)

        # All pixel values should be preserved, just rearranged
        original_sorted = np.sort(gradient_image.flatten())
        encrypted_sorted = np.sort(encrypted.flatten())

        np.testing.assert_array_equal(original_sorted, encrypted_sorted)

    def test_sensitivity_to_key(self, small_image):
        """Test that small changes in key produce different results."""
        height, width = small_image.shape[:2]

        encrypted1 = pyscramble.row_column_logistic_encrypt(small_image, width, height, 0.5)
        encrypted2 = pyscramble.row_column_logistic_encrypt(small_image, width, height, 0.5000001)

        # Even tiny key differences should produce different results
        assert not np.array_equal(encrypted1, encrypted2)

    def test_large_image(self, large_image):
        """Test with large image."""
        height, width = large_image.shape[:2]
        key = 0.618

        encrypted = pyscramble.row_column_logistic_encrypt(large_image, width, height, key)
        decrypted = pyscramble.row_column_logistic_decrypt(encrypted, width, height, key)

        np.testing.assert_array_equal(decrypted, large_image)

    def test_different_from_row_only(self, small_image):
        """Test that row-column scrambling differs from row-only scrambling."""
        height, width = small_image.shape[:2]
        key = 0.5

        row_only = pyscramble.row_logistic_encrypt(small_image, width, height, key)
        row_column = pyscramble.row_column_logistic_encrypt(small_image, width, height, key)

        # Row-column should produce different results than row-only
        assert not np.array_equal(row_only, row_column)

    def test_double_encryption_decryption(self, small_image):
        """Test double encryption and decryption with different keys."""
        height, width = small_image.shape[:2]
        key1 = 0.3
        key2 = 0.7

        # Double encrypt
        encrypted1 = pyscramble.row_column_logistic_encrypt(small_image, width, height, key1)
        encrypted2 = pyscramble.row_column_logistic_encrypt(encrypted1, width, height, key2)

        # Double decrypt (reverse order)
        decrypted1 = pyscramble.row_column_logistic_decrypt(encrypted2, width, height, key2)
        decrypted2 = pyscramble.row_column_logistic_decrypt(decrypted1, width, height, key1)

        np.testing.assert_array_equal(decrypted2, small_image)
