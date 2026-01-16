//! Row Column Logistic Scramble module
//!
//! Logistic map-based row-column scrambling algorithm

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Generate logistic sequence positions for a given starting x and length
fn generate_logistic_sequence(x_start: f64, n: usize) -> Vec<usize> {
    let mut logistic_arr: Vec<(f64, usize)> = Vec::with_capacity(n);
    let mut x = x_start;
    logistic_arr.push((x, 0));
    for i in 1..n {
        x = 3.9999999 * x * (1.0 - x);
        logistic_arr.push((x, i));
    }
    logistic_arr.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    logistic_arr.iter().map(|item| item.1).collect()
}

/// Pre-compute all x values for each row/column
fn compute_x_sequence(key: f64, count: usize, length: usize) -> Vec<f64> {
    let mut x_values = Vec::with_capacity(count);
    let mut x = key;
    for _ in 0..count {
        x_values.push(x);
        // Advance x through the sequence
        for _ in 0..length {
            x = 3.9999999 * x * (1.0 - x);
        }
    }
    x_values
}

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

    // Pre-compute x values for all rows and generate all position mappings
    let row_x_values = compute_x_sequence(key, h, w);
    let row_positions: Vec<Vec<usize>> = row_x_values
        .par_iter()
        .map(|&x| generate_logistic_sequence(x, w))
        .collect();

    // Step 1: Row scrambling (parallel)
    let total_pixels = w * h;
    let int_pixels_buf: Vec<i32> = (0..total_pixels)
        .into_par_iter()
        .map(|idx| {
            let i = idx % w;
            let j = idx / w;
            let positions = &row_positions[j];
            pixels[positions[i] + j * w]
        })
        .collect();

    // Pre-compute x values for all columns and generate all position mappings
    let col_x_values = compute_x_sequence(key, w, h);
    let col_positions: Vec<Vec<usize>> = col_x_values
        .par_iter()
        .map(|&x| generate_logistic_sequence(x, h))
        .collect();

    // Step 2: Column scrambling (parallel)
    let new_pixels: Vec<i32> = (0..total_pixels)
        .into_par_iter()
        .map(|idx| {
            let i = idx % w;
            let j = idx / w;
            let positions = &col_positions[i];
            int_pixels_buf[i + positions[j] * w]
        })
        .collect();

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
    let total_pixels = w * h;

    // Pre-compute x values for all columns and generate all position mappings
    let col_x_values = compute_x_sequence(key, w, h);
    let col_positions: Vec<Vec<usize>> = col_x_values
        .par_iter()
        .map(|&x| generate_logistic_sequence(x, h))
        .collect();

    // Step 1: Column descrambling (parallel)
    let mapping: Vec<(usize, i32)> = (0..total_pixels)
        .into_par_iter()
        .map(|idx| {
            let i = idx % w;
            let j = idx / w;
            let positions = &col_positions[i];
            (i + positions[j] * w, pixels[idx])
        })
        .collect();

    // Scatter results
    let mut int_pixels_buf = vec![0i32; total_pixels];
    for (dst, val) in mapping {
        int_pixels_buf[dst] = val;
    }

    // Pre-compute x values for all rows and generate all position mappings
    let row_x_values = compute_x_sequence(key, h, w);
    let row_positions: Vec<Vec<usize>> = row_x_values
        .par_iter()
        .map(|&x| generate_logistic_sequence(x, w))
        .collect();

    // Step 2: Row descrambling (parallel)
    let mapping: Vec<(usize, i32)> = (0..total_pixels)
        .into_par_iter()
        .map(|idx| {
            let i = idx % w;
            let j = idx / w;
            let positions = &row_positions[j];
            (positions[i] + j * w, int_pixels_buf[idx])
        })
        .collect();

    // Scatter results
    let mut new_pixels = vec![0i32; total_pixels];
    for (dst, val) in mapping {
        new_pixels[dst] = val;
    }

    PyArray1::from_vec(py, new_pixels)
}
