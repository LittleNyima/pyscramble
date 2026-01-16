//! Row Column Logistic Scramble module
//!
//! Logistic map-based row-column scrambling algorithm

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
pub fn row_column_logistic_encrypt<'py>(
    py: Python<'py>,
    int_pixels: PyReadonlyArray1<i32>,
    width: i32,
    height: i32,
    key: f64,
) -> Bound<'py, PyArray1<i32>> {
    let pixels = int_pixels.as_slice().unwrap();
    let w = width as usize;
    let h = height as usize;

    let mut new_pixels: Vec<i32> = pixels.to_vec();
    let mut int_pixels_buf: Vec<i32> = pixels.to_vec();

    let mut x = key;

    // Step 1: Row scrambling
    for j in 0..h {
        let offset = j * w;
        let mut logistic_arr: Vec<(f64, usize)> = Vec::with_capacity(w);
        logistic_arr.push((x, 0));
        for i in 1..w {
            x = 3.9999999 * x * (1.0 - x);
            logistic_arr.push((x, i));
        }
        x = 3.9999999 * x * (1.0 - x); // 更新x到下一个值

        logistic_arr.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let positions: Vec<usize> = logistic_arr.iter().map(|item| item.1).collect();

        for i in 0..w {
            int_pixels_buf[i + offset] = new_pixels[positions[i] + offset];
        }
    }

    // Step 2: Column scrambling
    x = key;
    for i in 0..w {
        let mut logistic_arr: Vec<(f64, usize)> = Vec::with_capacity(h);
        logistic_arr.push((x, 0));
        for ji in 1..h {
            x = 3.9999999 * x * (1.0 - x);
            logistic_arr.push((x, ji));
        }
        x = 3.9999999 * x * (1.0 - x);

        logistic_arr.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let positions: Vec<usize> = logistic_arr.iter().map(|item| item.1).collect();

        for j in 0..h {
            new_pixels[i + j * w] = int_pixels_buf[i + positions[j] * w];
        }
    }

    PyArray1::from_vec(py, new_pixels)
}

#[pyfunction]
pub fn row_column_logistic_decrypt<'py>(
    py: Python<'py>,
    int_pixels: PyReadonlyArray1<i32>,
    width: i32,
    height: i32,
    key: f64,
) -> Bound<'py, PyArray1<i32>> {
    let pixels = int_pixels.as_slice().unwrap();
    let w = width as usize;
    let h = height as usize;

    let mut new_pixels: Vec<i32> = pixels.to_vec();
    let mut int_pixels_buf: Vec<i32> = pixels.to_vec();

    // Step 1: Column descrambling
    let mut x = key;
    for i in 0..w {
        let mut logistic_arr: Vec<(f64, usize)> = Vec::with_capacity(h);
        logistic_arr.push((x, 0));
        for ji in 1..h {
            x = 3.9999999 * x * (1.0 - x);
            logistic_arr.push((x, ji));
        }
        x = 3.9999999 * x * (1.0 - x);

        logistic_arr.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let positions: Vec<usize> = logistic_arr.iter().map(|item| item.1).collect();

        for j in 0..h {
            int_pixels_buf[i + positions[j] * w] = new_pixels[i + j * w];
        }
    }

    // Step 2: Row descrambling
    x = key;
    for j in 0..h {
        let offset = j * w;
        let mut logistic_arr: Vec<(f64, usize)> = Vec::with_capacity(w);
        logistic_arr.push((x, 0));
        for i in 1..w {
            x = 3.9999999 * x * (1.0 - x);
            logistic_arr.push((x, i));
        }
        x = 3.9999999 * x * (1.0 - x);

        logistic_arr.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let positions: Vec<usize> = logistic_arr.iter().map(|item| item.1).collect();

        for i in 0..w {
            new_pixels[positions[i] + offset] = int_pixels_buf[i + offset];
        }
    }

    PyArray1::from_vec(py, new_pixels)
}
