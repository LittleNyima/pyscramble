//! Block MD5 Scramble module
//!
//! MD5 hash-based block scrambling algorithm

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::utils::shuffle_with_key;

#[pyfunction]
pub fn block_md5_encrypt<'py>(
    py: Python<'py>,
    int_pixels: PyReadonlyArray1<i32>,
    width: i32,
    height: i32,
    key: &str,
    x_block_count: i32,
    y_block_count: i32,
) -> (Bound<'py, PyArray1<i32>>, i32, i32) {
    let pixels = int_pixels.as_slice().unwrap();
    let w = width as usize;
    let h = height as usize;
    let xbc = x_block_count as usize;
    let ybc = y_block_count as usize;

    let x_array = shuffle_with_key(xbc, key);
    let y_array = shuffle_with_key(ybc, key);

    // Calculate new dimensions
    let new_width = if w % xbc > 0 { w + xbc - w % xbc } else { w };
    let new_height = if h % ybc > 0 { h + ybc - h % ybc } else { h };

    let block_width = new_width / xbc;
    let block_height = new_height / ybc;

    let mut new_pixels = vec![0i32; new_width * new_height];

    for i in 0..new_width {
        for j in 0..new_height {
            let mut n = j;
            let mut m = (x_array[(n / block_height) % xbc] as usize * block_width + i) % new_width;
            m = x_array[m / block_width] as usize * block_width + m % block_width;
            n = (y_array[m / block_width % ybc] as usize * block_height + n) % new_height;
            n = y_array[n / block_height] as usize * block_height + n % block_height;
            new_pixels[i + j * new_width] = pixels[(m % w) + (n % h) * w];
        }
    }

    (
        PyArray1::from_vec(py, new_pixels),
        new_width as i32,
        new_height as i32,
    )
}

#[pyfunction]
pub fn block_md5_decrypt<'py>(
    py: Python<'py>,
    int_pixels: PyReadonlyArray1<i32>,
    width: i32,
    height: i32,
    key: &str,
    x_block_count: i32,
    y_block_count: i32,
) -> Bound<'py, PyArray1<i32>> {
    let pixels = int_pixels.as_slice().unwrap();
    let w = width as usize;
    let h = height as usize;
    let xbc = x_block_count as usize;
    let ybc = y_block_count as usize;

    let x_array = shuffle_with_key(xbc, key);
    let y_array = shuffle_with_key(ybc, key);

    let block_width = w / xbc;
    let block_height = h / ybc;

    let mut new_pixels = vec![0i32; w * h];

    for i in 0..w {
        for j in 0..h {
            let mut n = j;
            let mut m = (x_array[(n / block_height) % xbc] as usize * block_width + i) % w;
            m = x_array[m / block_width] as usize * block_width + m % block_width;
            n = (y_array[m / block_width % ybc] as usize * block_height + n) % h;
            n = y_array[n / block_height] as usize * block_height + n % block_height;
            new_pixels[m + n * w] = pixels[(i % w) + (j % h) * w];
        }
    }

    PyArray1::from_vec(py, new_pixels)
}
