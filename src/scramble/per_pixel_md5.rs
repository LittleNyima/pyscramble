//! Pixel MD5 Scramble module
//!
//! MD5 hash-based per-pixel scrambling algorithm

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::utils::shuffle_with_key;

#[pyfunction]
pub fn per_pixel_md5_encrypt<'py>(
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
    let y_array = shuffle_with_key(h, key);

    let mut new_pixels = vec![0i32; w * h];

    for i in 0..w {
        for j in 0..h {
            let m = x_array[(x_array[j % w] as usize + i) % w] as usize;
            let n = y_array[(y_array[m % h] as usize + j) % h] as usize;
            new_pixels[i + j * w] = pixels[m + n * w];
        }
    }

    PyArray1::from_vec(py, new_pixels)
}

#[pyfunction]
pub fn per_pixel_md5_decrypt<'py>(
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
    let y_array = shuffle_with_key(h, key);

    let mut new_pixels = vec![0i32; w * h];

    for i in 0..w {
        for j in 0..h {
            let m = x_array[(x_array[j % w] as usize + i) % w] as usize;
            let n = y_array[(y_array[m % h] as usize + j) % h] as usize;
            new_pixels[m + n * w] = pixels[i + j * w];
        }
    }

    PyArray1::from_vec(py, new_pixels)
}
