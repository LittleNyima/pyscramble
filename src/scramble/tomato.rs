//! Tomato Scramble module
//!
//! Pixel scrambling algorithm based on Gilbert 2D space-filling curve

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::utils::gilbert2d;

#[pyfunction]
pub fn tomato_scramble_encrypt<'py>(
    py: Python<'py>,
    int_pixels: PyReadonlyArray1<i32>,
    width: i32,
    height: i32,
    key: f64,
) -> Bound<'py, PyArray1<i32>> {
    let pixels = int_pixels.as_slice().unwrap();
    let pixel_count = (width * height) as usize;

    let offset =
        (((5.0_f64.sqrt() - 1.0) / 2.0 * pixel_count as f64 * key).round() as usize) % pixel_count;
    let positions = gilbert2d(width, height);

    let loop_position = pixel_count - offset;
    let mut new_pixels = vec![0i32; pixel_count];

    for i in 0..loop_position {
        new_pixels[positions[i + offset] as usize] = pixels[positions[i] as usize];
    }
    for i in loop_position..pixel_count {
        new_pixels[positions[i - loop_position] as usize] = pixels[positions[i] as usize];
    }

    PyArray1::from_vec(py, new_pixels)
}

#[pyfunction]
pub fn tomato_scramble_decrypt<'py>(
    py: Python<'py>,
    int_pixels: PyReadonlyArray1<i32>,
    width: i32,
    height: i32,
    key: f64,
) -> Bound<'py, PyArray1<i32>> {
    let pixels = int_pixels.as_slice().unwrap();
    let pixel_count = (width * height) as usize;

    let offset =
        (((5.0_f64.sqrt() - 1.0) / 2.0 * pixel_count as f64 * key).round() as usize) % pixel_count;
    let positions = gilbert2d(width, height);

    let loop_position = pixel_count - offset;
    let mut new_pixels = vec![0i32; pixel_count];

    for i in 0..loop_position {
        new_pixels[positions[i] as usize] = pixels[positions[i + offset] as usize];
    }
    for i in loop_position..pixel_count {
        new_pixels[positions[i] as usize] = pixels[positions[i - loop_position] as usize];
    }

    PyArray1::from_vec(py, new_pixels)
}
