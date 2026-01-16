"""
Tests for Block MD5 Scramble algorithm.

MD5 hash-based block scrambling.
"""

import numpy as np

import pyscramble


class TestBlockMD5:
    """Test suite for block_md5_encrypt and block_md5_decrypt."""

    def test_encrypt_decrypt_roundtrip(self, medium_image):
        """Test that encrypt followed by decrypt returns original image."""
        height, width = medium_image.shape[:2]
        key = "test_key"
        x_block_count = 8
        y_block_count = 8

        encrypted, new_width, new_height = pyscramble.block_md5_encrypt(
            medium_image, width, height, key, x_block_count, y_block_count
        )
        decrypted, _, _ = pyscramble.block_md5_decrypt(
            encrypted, new_width, new_height, key, x_block_count, y_block_count
        )

        # Crop to original size for comparison
        decrypted_cropped = decrypted[:height, :width]
        np.testing.assert_array_equal(decrypted_cropped, medium_image)

    def test_encrypt_changes_image(self, medium_image):
        """Test that encryption actually changes the image."""
        height, width = medium_image.shape[:2]
        key = "test_key"

        encrypted, _, _ = pyscramble.block_md5_encrypt(medium_image, width, height, key)

        # Compare only the overlapping region
        min_height = min(height, encrypted.shape[0])
        min_width = min(width, encrypted.shape[1])

        assert not np.array_equal(encrypted[:min_height, :min_width], medium_image[:min_height, :min_width])

    def test_different_keys_produce_different_results(self, medium_image):
        """Test that different keys produce different encrypted images."""
        height, width = medium_image.shape[:2]

        encrypted1, _, _ = pyscramble.block_md5_encrypt(medium_image, width, height, "key1")
        encrypted2, _, _ = pyscramble.block_md5_encrypt(medium_image, width, height, "key2")

        assert not np.array_equal(encrypted1, encrypted2)

    def test_same_key_produces_same_result(self, medium_image):
        """Test that same key produces identical encrypted images."""
        height, width = medium_image.shape[:2]
        key = "consistent_key"

        encrypted1, w1, h1 = pyscramble.block_md5_encrypt(medium_image, width, height, key)
        encrypted2, w2, h2 = pyscramble.block_md5_encrypt(medium_image, width, height, key)

        assert w1 == w2
        assert h1 == h2
        np.testing.assert_array_equal(encrypted1, encrypted2)

    def test_output_dimensions_match_returned_values(self, medium_image):
        """Test that output dimensions match returned width and height."""
        height, width = medium_image.shape[:2]
        key = "test_key"

        encrypted, new_width, new_height = pyscramble.block_md5_encrypt(medium_image, width, height, key)

        assert encrypted.shape[0] == new_height
        assert encrypted.shape[1] == new_width
        assert encrypted.shape[2] == 4  # RGBA channels

    def test_padding_for_non_divisible_dimensions(self):
        """Test that images are padded when dimensions are not divisible by block count."""
        # Create image with dimensions not divisible by block count
        np.random.seed(42)
        pixels = np.random.randint(0, 256, (50, 50, 4), dtype=np.uint8)
        height, width = 50, 50
        key = "padding_test"
        x_block_count = 8
        y_block_count = 8

        encrypted, new_width, new_height = pyscramble.block_md5_encrypt(
            pixels, width, height, key, x_block_count, y_block_count
        )

        # New dimensions should be >= original and divisible by block count
        assert new_width >= width
        assert new_height >= height
        assert new_width % x_block_count == 0
        assert new_height % y_block_count == 0

    def test_default_block_counts(self, large_image):
        """Test with default block counts (32x32)."""
        height, width = large_image.shape[:2]
        key = "default_test"

        encrypted, new_width, new_height = pyscramble.block_md5_encrypt(large_image, width, height, key)
        decrypted, _, _ = pyscramble.block_md5_decrypt(encrypted, new_width, new_height, key)

        # Crop to original size for comparison
        decrypted_cropped = decrypted[:height, :width]
        np.testing.assert_array_equal(decrypted_cropped, large_image)

    def test_custom_block_counts(self, medium_image):
        """Test with custom block counts."""
        height, width = medium_image.shape[:2]
        key = "custom_test"
        x_block_count = 4
        y_block_count = 16

        encrypted, new_width, new_height = pyscramble.block_md5_encrypt(
            medium_image, width, height, key, x_block_count, y_block_count
        )
        decrypted, _, _ = pyscramble.block_md5_decrypt(
            encrypted, new_width, new_height, key, x_block_count, y_block_count
        )

        # Crop to original size for comparison
        decrypted_cropped = decrypted[:height, :width]
        np.testing.assert_array_equal(decrypted_cropped, medium_image)

    def test_wrong_key_fails_decrypt(self, medium_image):
        """Test that wrong key does not correctly decrypt."""
        height, width = medium_image.shape[:2]

        encrypted, new_width, new_height = pyscramble.block_md5_encrypt(medium_image, width, height, "correct_key")
        decrypted, _, _ = pyscramble.block_md5_decrypt(encrypted, new_width, new_height, "wrong_key")

        decrypted_cropped = decrypted[:height, :width]
        assert not np.array_equal(decrypted_cropped, medium_image)

    def test_wrong_block_count_fails_decrypt(self, medium_image):
        """Test that wrong block count does not correctly decrypt."""
        height, width = medium_image.shape[:2]
        key = "test_key"

        encrypted, new_width, new_height = pyscramble.block_md5_encrypt(medium_image, width, height, key, 8, 8)
        decrypted, _, _ = pyscramble.block_md5_decrypt(encrypted, new_width, new_height, key, 4, 4)

        decrypted_cropped = decrypted[:height, :width]
        assert not np.array_equal(decrypted_cropped, medium_image)

    def test_empty_key(self, medium_image):
        """Test with empty string key."""
        height, width = medium_image.shape[:2]
        key = ""
        x_block_count = 8
        y_block_count = 8

        encrypted, new_width, new_height = pyscramble.block_md5_encrypt(
            medium_image, width, height, key, x_block_count, y_block_count
        )
        decrypted, _, _ = pyscramble.block_md5_decrypt(
            encrypted, new_width, new_height, key, x_block_count, y_block_count
        )

        decrypted_cropped = decrypted[:height, :width]
        np.testing.assert_array_equal(decrypted_cropped, medium_image)

    def test_unicode_key(self, medium_image):
        """Test with unicode string key."""
        height, width = medium_image.shape[:2]
        key = "块加密测试🧱"
        x_block_count = 8
        y_block_count = 8

        encrypted, new_width, new_height = pyscramble.block_md5_encrypt(
            medium_image, width, height, key, x_block_count, y_block_count
        )
        decrypted, _, _ = pyscramble.block_md5_decrypt(
            encrypted, new_width, new_height, key, x_block_count, y_block_count
        )

        decrypted_cropped = decrypted[:height, :width]
        np.testing.assert_array_equal(decrypted_cropped, medium_image)

    def test_non_square_image(self, non_square_image):
        """Test with non-square image dimensions."""
        height, width = non_square_image.shape[:2]
        key = "non_square_key"
        x_block_count = 4
        y_block_count = 8

        encrypted, new_width, new_height = pyscramble.block_md5_encrypt(
            non_square_image, width, height, key, x_block_count, y_block_count
        )
        decrypted, _, _ = pyscramble.block_md5_decrypt(
            encrypted, new_width, new_height, key, x_block_count, y_block_count
        )

        decrypted_cropped = decrypted[:height, :width]
        np.testing.assert_array_equal(decrypted_cropped, non_square_image)
