//! Pixel conversion module
//!
//! Provides conversion between RGBA pixel arrays and integer arrays

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Convert RGBA pixel array to integer array
/// pixels shape: (height, width, 4) -> flatten to (height*width,) int32
/// Format: A<<24 | R<<16 | G<<8 | B
#[pyfunction]
pub fn pixels_to_int<'py>(
    py: Python<'py>,
    pixels: PyReadonlyArray3<u8>,
) -> Bound<'py, PyArray1<i32>> {
    let shape = pixels.shape();
    let height = shape[0];
    let width = shape[1];
    let pixel_count = height * width;

    let data = pixels.as_slice().unwrap();
    let mut result = Vec::with_capacity(pixel_count);

    for i in 0..pixel_count {
        let base = i * 4;
        let r = data[base] as i32;
        let g = data[base + 1] as i32;
        let b = data[base + 2] as i32;
        let a = data[base + 3] as i32;
        result.push((a << 24) | (r << 16) | (g << 8) | b);
    }

    PyArray1::from_vec(py, result)
}

/// Convert integer array back to RGBA pixel array
/// Returns flattened u8 array, Python side needs to reshape
#[pyfunction]
pub fn int_to_pixels<'py>(
    py: Python<'py>,
    int_pixels: PyReadonlyArray1<i32>,
) -> Bound<'py, PyArray1<u8>> {
    let data = int_pixels.as_slice().unwrap();
    let pixel_count = data.len();
    let mut result = Vec::with_capacity(pixel_count * 4);

    for &pixel in data {
        let r = ((pixel >> 16) & 0xFF) as u8;
        let g = ((pixel >> 8) & 0xFF) as u8;
        let b = (pixel & 0xFF) as u8;
        let a = ((pixel >> 24) & 0xFF) as u8;
        result.push(r);
        result.push(g);
        result.push(b);
        result.push(a);
    }

    PyArray1::from_vec(py, result)
}
