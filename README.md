# pyscramble

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-2024-orange.svg)](https://www.rust-lang.org/)

High-performance image pixel scrambling library written in Rust with Python bindings.

**pyscramble** provides various image pixel scramble/descramble algorithms implemented in Rust for maximum performance, exposed to Python via PyO3. It's designed for applications requiring reversible image obfuscation.

## ✨ Features

- 🚀 **High Performance** - Core algorithms implemented in Rust with parallel processing via Rayon
- 🐍 **Python-First API** - Clean, Pythonic interface with NumPy array support
- 🔐 **Multiple Algorithms** - 6 different scrambling algorithms for various use cases
- 🔄 **Reversible** - All scrambling operations are fully reversible with the correct key
- 📐 **Flexible Dimensions** - Works with images of any size (non-square images supported)

## 📦 Installation

### From PyPI

```bash
pip install pyscramble
```

### From Source (requires Rust toolchain)

```bash
# Clone the repository
git clone https://github.com/LittleNyima/pyscramble.git
cd pyscramble

# Install with maturin
pip install maturin
maturin develop --release
```

### Requirements

- Python >= 3.8
- NumPy
- Rust toolchain (for building from source)

## 🚀 Quick Start

```python
import numpy as np
from PIL import Image
import pyscramble

# Load an image as RGBA
image = Image.open("input.png").convert("RGBA")
pixels = np.array(image, dtype=np.uint8)

# Encrypt with Tomato Scramble
key = 1.0
encrypted = pyscramble.tomato_scramble_encrypt(pixels, key)

# Save encrypted image
Image.fromarray(encrypted).save("encrypted.png")

# Decrypt
decrypted = pyscramble.tomato_scramble_decrypt(encrypted, key)

# Verify
assert np.array_equal(decrypted, pixels)
```

## 🔧 API Reference

### 1. Tomato Scramble

Based on the **Gilbert 2D space-filling curve**. Provides a unique scrambling pattern based on a floating-point key.

```python
import pyscramble

# Encrypt
encrypted = pyscramble.tomato_scramble_encrypt(pixels, key=1.0)

# Decrypt
decrypted = pyscramble.tomato_scramble_decrypt(encrypted, key=1.0)
```

**Parameters:**

- `pixels`: Input RGBA pixel array with shape `(height, width, 4)`
- `key`: Encryption key (float, default: 1.0)

### 2. Per-Pixel MD5 Scramble

Uses **MD5 hash-based per-pixel scrambling**. Each pixel position is shuffled based on MD5 hash values.

```python
import pyscramble

# Encrypt
encrypted = pyscramble.per_pixel_md5_encrypt(pixels, key="secret_key")

# Decrypt
decrypted = pyscramble.per_pixel_md5_decrypt(encrypted, key="secret_key")
```

**Parameters:**

- `pixels`: Input RGBA pixel array with shape `(height, width, 4)`
- `key`: Encryption key (string)

### 3. Row MD5 Scramble

Uses **MD5 hash-based row pixel scrambling**. Pixels within each row are shuffled based on MD5 hash values.

```python
import pyscramble

# Encrypt
encrypted = pyscramble.row_md5_encrypt(pixels, key="secret_key")

# Decrypt
decrypted = pyscramble.row_md5_decrypt(encrypted, key="secret_key")
```

**Parameters:**

- `pixels`: Input RGBA pixel array with shape `(height, width, 4)`
- `key`: Encryption key (string)

### 4. Block MD5 Scramble

Uses **MD5 hash-based block scrambling**. The image is divided into blocks which are then shuffled. Images are automatically padded to fit the block grid.

```python
import pyscramble

# Encrypt (returns padded dimensions)
height, width = pixels.shape[:2]
encrypted, new_width, new_height = pyscramble.block_md5_encrypt(
    pixels, width, height,
    key="secret_key",
    x_block_count=8,
    y_block_count=8
)

# Decrypt
decrypted, _, _ = pyscramble.block_md5_decrypt(
    encrypted, new_width, new_height,
    key="secret_key",
    x_block_count=8,
    y_block_count=8
)

# Crop back to original size
decrypted_original = decrypted[:height, :width]
```

**Parameters:**

- `pixels`: Input RGBA pixel array with shape `(height, width, 4)`
- `width`: Image width
- `height`: Image height
- `key`: Encryption key (string)
- `x_block_count`: Number of blocks in x direction (default: 32)
- `y_block_count`: Number of blocks in y direction (default: 32)

**Returns:** Tuple of `(encrypted_pixels, new_width, new_height)`

### 5. Row Logistic Scramble

Uses the **logistic map** for row-based scrambling. The chaotic nature of the logistic map provides good scrambling properties.

```python
import pyscramble

# Encrypt
encrypted = pyscramble.row_logistic_encrypt(pixels, key=0.5)

# Decrypt
decrypted = pyscramble.row_logistic_decrypt(encrypted, key=0.5)
```

**Parameters:**

- `pixels`: Input RGBA pixel array with shape `(height, width, 4)`
- `key`: Encryption key (float, should be in range (0, 1))

### 6. Row-Column Logistic Scramble

Uses the **logistic map** for both row and column scrambling, providing a more thorough scrambling effect.

```python
import pyscramble

# Encrypt
encrypted = pyscramble.row_column_logistic_encrypt(pixels, key=0.5)

# Decrypt
decrypted = pyscramble.row_column_logistic_decrypt(encrypted, key=0.5)
```

**Parameters:**

- `pixels`: Input RGBA pixel array with shape `(height, width, 4)`
- `key`: Encryption key (float, should be in range (0, 1))

## ⚠️ Important Notes

1. **Key Sensitivity**: Using the wrong key for decryption will NOT restore the original image
2. **Block Count Consistency**: For `block_md5` functions, use the same block counts for encryption and decryption
3. **Logistic Key Range**: For logistic-based algorithms, keep the key in range `(0, 1)` for best results
4. **Image Format**: Convert images to RGBA format before processing

## 🧪 Running Tests

```bash
# Install test dependencies
pip install pytest

# Run tests
pytest tests/ -v
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PyO3](https://pyo3.rs/) - Rust bindings for Python
- [Rayon](https://github.com/rayon-rs/rayon) - Data parallelism library for Rust
- [Maturin](https://maturin.rs/) - Build and publish Rust-based Python packages
