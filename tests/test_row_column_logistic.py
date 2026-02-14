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
        key = 0.5

        encrypted = pyscramble.row_column_logistic_encrypt(small_image, key)
        decrypted = pyscramble.row_column_logistic_decrypt(encrypted, key)

        np.testing.assert_array_equal(decrypted, small_image)

    def test_encrypt_changes_image(self, small_image):
        """Test that encryption actually changes the image."""
        key = 0.5

        encrypted = pyscramble.row_column_logistic_encrypt(small_image, key)

        assert not np.array_equal(encrypted, small_image)

    def test_different_keys_produce_different_results(self, small_image):
        """Test that different keys produce different encrypted images."""
        encrypted1 = pyscramble.row_column_logistic_encrypt(small_image, 0.3)
        encrypted2 = pyscramble.row_column_logistic_encrypt(small_image, 0.7)

        assert not np.array_equal(encrypted1, encrypted2)

    def test_same_key_produces_same_result(self, small_image):
        """Test that same key produces identical encrypted images."""
        key = 0.456

        encrypted1 = pyscramble.row_column_logistic_encrypt(small_image, key)
        encrypted2 = pyscramble.row_column_logistic_encrypt(small_image, key)

        np.testing.assert_array_equal(encrypted1, encrypted2)

    def test_output_shape_preserved(self, small_image):
        """Test that output shape matches input shape."""
        key = 0.5

        encrypted = pyscramble.row_column_logistic_encrypt(small_image, key)

        assert encrypted.shape == small_image.shape

    def test_medium_image(self, medium_image):
        """Test with medium-sized image."""
        key = 0.123

        encrypted = pyscramble.row_column_logistic_encrypt(medium_image, key)
        decrypted = pyscramble.row_column_logistic_decrypt(encrypted, key)

        np.testing.assert_array_equal(decrypted, medium_image)

    def test_non_square_image(self, non_square_image):
        """Test with non-square image dimensions."""
        key = 0.789

        encrypted = pyscramble.row_column_logistic_encrypt(non_square_image, key)
        decrypted = pyscramble.row_column_logistic_decrypt(encrypted, key)

        np.testing.assert_array_equal(decrypted, non_square_image)

    def test_key_near_zero(self, small_image):
        """Test with key value close to 0."""
        key = 0.001

        encrypted = pyscramble.row_column_logistic_encrypt(small_image, key)
        decrypted = pyscramble.row_column_logistic_decrypt(encrypted, key)

        np.testing.assert_array_equal(decrypted, small_image)

    def test_key_near_one(self, small_image):
        """Test with key value close to 1."""
        key = 0.999

        encrypted = pyscramble.row_column_logistic_encrypt(small_image, key)
        decrypted = pyscramble.row_column_logistic_decrypt(encrypted, key)

        np.testing.assert_array_equal(decrypted, small_image)

    def test_wrong_key_fails_decrypt(self, small_image):
        """Test that wrong key does not correctly decrypt."""
        encrypted = pyscramble.row_column_logistic_encrypt(small_image, 0.5)
        decrypted = pyscramble.row_column_logistic_decrypt(encrypted, 0.6)

        assert not np.array_equal(decrypted, small_image)

    def test_pixel_values_preserved(self, gradient_image):
        """Test that pixel values are preserved (just rearranged)."""
        key = 0.5

        encrypted = pyscramble.row_column_logistic_encrypt(gradient_image, key)

        # All pixel values should be preserved, just rearranged
        original_sorted = np.sort(gradient_image.flatten())
        encrypted_sorted = np.sort(encrypted.flatten())

        np.testing.assert_array_equal(original_sorted, encrypted_sorted)

    def test_sensitivity_to_key(self, small_image):
        """Test that small changes in key produce different results."""
        encrypted1 = pyscramble.row_column_logistic_encrypt(small_image, 0.5)
        encrypted2 = pyscramble.row_column_logistic_encrypt(small_image, 0.5000001)

        # Even tiny key differences should produce different results
        assert not np.array_equal(encrypted1, encrypted2)

    def test_large_image(self, large_image):
        """Test with large image."""
        key = 0.618

        encrypted = pyscramble.row_column_logistic_encrypt(large_image, key)
        decrypted = pyscramble.row_column_logistic_decrypt(encrypted, key)

        np.testing.assert_array_equal(decrypted, large_image)

    def test_different_from_row_only(self, small_image):
        """Test that row-column scrambling differs from row-only scrambling."""
        key = 0.5

        row_only = pyscramble.row_logistic_encrypt(small_image, key)
        row_column = pyscramble.row_column_logistic_encrypt(small_image, key)

        # Row-column should produce different results than row-only
        assert not np.array_equal(row_only, row_column)

    def test_double_encryption_decryption(self, small_image):
        """Test double encryption and decryption with different keys."""
        key1 = 0.3
        key2 = 0.7

        # Double encrypt
        encrypted1 = pyscramble.row_column_logistic_encrypt(small_image, key1)
        encrypted2 = pyscramble.row_column_logistic_encrypt(encrypted1, key2)

        # Double decrypt (reverse order)
        decrypted1 = pyscramble.row_column_logistic_decrypt(encrypted2, key2)
        decrypted2 = pyscramble.row_column_logistic_decrypt(decrypted1, key1)

        np.testing.assert_array_equal(decrypted2, small_image)
