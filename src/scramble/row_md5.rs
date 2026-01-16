//! Row MD5 Scramble module
//!
//! MD5 hash-based row pixel scrambling algorithm

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::utils::shuffle_with_key;

#[pyfunction]
pub fn row_md5_encrypt<'py>(
    py: Python<'py>,
    int_pixels: PyReadonlyArray1<i32>,
    width: i32,
    height: i32,
    key: &str,
) -> Bound<'py, PyArray1<i32>> {
    let pixels = int_pixels.as_slice().unwrap();
    let w = width as usize;
    let h = height as usize;

    let x_array = shuffle_with_key(w, key);

    let total_pixels = w * h;
    let new_pixels: Vec<i32> = (0..total_pixels)
        .into_par_iter()
        .map(|idx| {
            let i = idx % w;
            let j = idx / w;
            let m = x_array[(x_array[j % w] as usize + i) % w] as usize;
            pixels[m + j * w]
        })
        .collect();

    PyArray1::from_vec(py, new_pixels)
}

#[pyfunction]
pub fn row_md5_decrypt<'py>(
    py: Python<'py>,
    int_pixels: PyReadonlyArray1<i32>,
    width: i32,
    height: i32,
    key: &str,
) -> Bound<'py, PyArray1<i32>> {
    let pixels = int_pixels.as_slice().unwrap();
    let w = width as usize;
    let h = height as usize;

    let x_array = shuffle_with_key(w, key);

    // Build mapping in parallel
    let total_pixels = w * h;
    let mapping: Vec<(usize, i32)> = (0..total_pixels)
        .into_par_iter()
        .map(|idx| {
            let i = idx % w;
            let j = idx / w;
            let m = x_array[(x_array[j % w] as usize + i) % w] as usize;
            (m + j * w, pixels[idx])
        })
        .collect();

    // Scatter results
    let mut new_pixels = vec![0i32; total_pixels];
    for (dst, val) in mapping {
        new_pixels[dst] = val;
    }

    PyArray1::from_vec(py, new_pixels)
}
