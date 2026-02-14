"""
Tests for Tomato Scramble algorithm.

Based on Gilbert 2D space-filling curve.
"""

import numpy as np

import pyscramble


class TestTomatoScramble:
    """Test suite for tomato_scramble_encrypt and tomato_scramble_decrypt."""

    def test_encrypt_decrypt_roundtrip(self, small_image):
        """Test that encrypt followed by decrypt returns original image."""
        key = 1.0

        encrypted = pyscramble.tomato_scramble_encrypt(small_image, key)
        decrypted = pyscramble.tomato_scramble_decrypt(encrypted, key)

        np.testing.assert_array_equal(decrypted, small_image)

    def test_encrypt_changes_image(self, small_image):
        """Test that encryption actually changes the image."""
        key = 1.0

        encrypted = pyscramble.tomato_scramble_encrypt(small_image, key)

        assert not np.array_equal(encrypted, small_image)

    def test_different_keys_produce_different_results(self, small_image):
        """Test that different keys produce different encrypted images."""
        encrypted1 = pyscramble.tomato_scramble_encrypt(small_image, 1.0)
        encrypted2 = pyscramble.tomato_scramble_encrypt(small_image, 2.0)

        assert not np.array_equal(encrypted1, encrypted2)

    def test_same_key_produces_same_result(self, small_image):
        """Test that same key produces identical encrypted images."""
        key = 1.5

        encrypted1 = pyscramble.tomato_scramble_encrypt(small_image, key)
        encrypted2 = pyscramble.tomato_scramble_encrypt(small_image, key)

        np.testing.assert_array_equal(encrypted1, encrypted2)

    def test_output_shape_preserved(self, small_image):
        """Test that output shape matches input shape."""
        key = 1.0

        encrypted = pyscramble.tomato_scramble_encrypt(small_image, key)

        assert encrypted.shape == small_image.shape

    def test_medium_image(self, medium_image):
        """Test with medium-sized image."""
        key = 0.5

        encrypted = pyscramble.tomato_scramble_encrypt(medium_image, key)
        decrypted = pyscramble.tomato_scramble_decrypt(encrypted, key)

        np.testing.assert_array_equal(decrypted, medium_image)

    def test_non_square_image(self, non_square_image):
        """Test with non-square image dimensions."""
        key = 1.0

        encrypted = pyscramble.tomato_scramble_encrypt(non_square_image, key)
        decrypted = pyscramble.tomato_scramble_decrypt(encrypted, key)

        np.testing.assert_array_equal(decrypted, non_square_image)

    def test_default_key(self, small_image):
        """Test that default key value works."""
        # Using default key
        encrypted = pyscramble.tomato_scramble_encrypt(small_image)
        decrypted = pyscramble.tomato_scramble_decrypt(encrypted)

        np.testing.assert_array_equal(decrypted, small_image)

    def test_wrong_key_fails_decrypt(self, small_image):
        """Test that wrong key does not correctly decrypt."""
        encrypted = pyscramble.tomato_scramble_encrypt(small_image, 1.0)
        decrypted = pyscramble.tomato_scramble_decrypt(encrypted, 2.0)

        assert not np.array_equal(decrypted, small_image)

    def test_pixel_values_preserved(self, gradient_image):
        """Test that pixel values are preserved (just rearranged)."""
        key = 1.0

        encrypted = pyscramble.tomato_scramble_encrypt(gradient_image, key)

        # All pixel values should be preserved, just rearranged
        original_sorted = np.sort(gradient_image.flatten())
        encrypted_sorted = np.sort(encrypted.flatten())

        np.testing.assert_array_equal(original_sorted, encrypted_sorted)
