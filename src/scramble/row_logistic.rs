//! Row Logistic Scramble module
//!
//! Logistic map-based row scrambling algorithm

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::utils::generate_logistic_positions;

#[pyfunction]
pub fn row_logistic_encrypt<'py>(
    py: Python<'py>,
    int_pixels: PyReadonlyArray1<i32>,
    width: i32,
    height: i32,
    key: f64,
) -> Bound<'py, PyArray1<i32>> {
    let pixels = int_pixels.as_slice().unwrap();
    let w = width as usize;
    let h = height as usize;

    let positions = generate_logistic_positions(key, w);

    let pixel_count = w * h;
    let mut new_pixels = vec![0i32; pixel_count];
    let offset = (h - 1) * w;

    for i in 0..w {
        let m = positions[i] as usize;
        let mut j = offset as i32;
        while j >= 0 {
            let ju = j as usize;
            new_pixels[i + ju] = pixels[m + ju];
            j -= w as i32;
        }
    }

    PyArray1::from_vec(py, new_pixels)
}

#[pyfunction]
pub fn row_logistic_decrypt<'py>(
    py: Python<'py>,
    int_pixels: PyReadonlyArray1<i32>,
    width: i32,
    height: i32,
    key: f64,
) -> Bound<'py, PyArray1<i32>> {
    let pixels = int_pixels.as_slice().unwrap();
    let w = width as usize;
    let h = height as usize;

    let positions = generate_logistic_positions(key, w);

    let pixel_count = w * h;
    let mut new_pixels = vec![0i32; pixel_count];
    let offset = (h - 1) * w;

    for i in 0..w {
        let m = positions[i] as usize;
        let mut j = offset as i32;
        while j >= 0 {
            let ju = j as usize;
            new_pixels[m + ju] = pixels[i + ju];
            j -= w as i32;
        }
    }

    PyArray1::from_vec(py, new_pixels)
}
