//! pyscramble - High-performance image pixel scrambling algorithms
//!
//! Rust implementation of various image pixel scramble/descramble algorithms:
//! - Tomato Scramble: Based on Gilbert 2D space-filling curve
//! - Pixel MD5 Scramble: MD5 hash-based per-pixel scrambling
//! - Row MD5 Scramble: MD5 hash-based row pixel scrambling
//! - Block MD5 Scramble: MD5 hash-based block scrambling
//! - Row Logistic Scramble: Logistic map-based row scrambling
//! - Row Column Logistic Scramble: Logistic map-based row-column scrambling

use pyo3::prelude::*;

// Internal modules
mod pixels;
mod scramble;
mod utils;

// Re-export pyfunctions
use pixels::{int_to_pixels, pixels_to_int};
use scramble::{
    block_md5_decrypt, block_md5_encrypt, per_pixel_md5_decrypt, per_pixel_md5_encrypt,
    row_column_logistic_decrypt, row_column_logistic_encrypt, row_logistic_decrypt,
    row_logistic_encrypt, row_md5_decrypt, row_md5_encrypt, tomato_scramble_decrypt,
    tomato_scramble_encrypt,
};

/// Python module entry point
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Pixel conversion functions
    m.add_function(wrap_pyfunction!(pixels_to_int, m)?)?;
    m.add_function(wrap_pyfunction!(int_to_pixels, m)?)?;

    // Tomato Scramble
    m.add_function(wrap_pyfunction!(tomato_scramble_encrypt, m)?)?;
    m.add_function(wrap_pyfunction!(tomato_scramble_decrypt, m)?)?;

    // Pixel MD5 Scramble
    m.add_function(wrap_pyfunction!(per_pixel_md5_encrypt, m)?)?;
    m.add_function(wrap_pyfunction!(per_pixel_md5_decrypt, m)?)?;

    // Row MD5 Scramble
    m.add_function(wrap_pyfunction!(row_md5_encrypt, m)?)?;
    m.add_function(wrap_pyfunction!(row_md5_decrypt, m)?)?;

    // Block MD5 Scramble
    m.add_function(wrap_pyfunction!(block_md5_encrypt, m)?)?;
    m.add_function(wrap_pyfunction!(block_md5_decrypt, m)?)?;

    // Row Logistic Scramble
    m.add_function(wrap_pyfunction!(row_logistic_encrypt, m)?)?;
    m.add_function(wrap_pyfunction!(row_logistic_decrypt, m)?)?;

    // Row Column Logistic Scramble
    m.add_function(wrap_pyfunction!(row_column_logistic_encrypt, m)?)?;
    m.add_function(wrap_pyfunction!(row_column_logistic_decrypt, m)?)?;

    Ok(())
}
