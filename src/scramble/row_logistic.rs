//! Row Logistic Scramble module
//!
//! Logistic map-based row scrambling algorithm

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

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

    let total_pixels = w * h;
    let new_pixels: Vec<i32> = (0..total_pixels)
        .into_par_iter()
        .map(|idx| {
            let i = idx % w;
            let row = idx / w;
            let m = positions[i] as usize;
            pixels[m + row * w]
        })
        .collect();

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

    // Build mapping in parallel
    let total_pixels = w * h;
    let mapping: Vec<(usize, i32)> = (0..total_pixels)
        .into_par_iter()
        .map(|idx| {
            let i = idx % w;
            let row = idx / w;
            let m = positions[i] as usize;
            (m + row * w, pixels[idx])
        })
        .collect();

    // Scatter results
    let mut new_pixels = vec![0i32; total_pixels];
    for (dst, val) in mapping {
        new_pixels[dst] = val;
    }

    PyArray1::from_vec(py, new_pixels)
}
