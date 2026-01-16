//! Utility functions module
//!
//! Contains Logistic chaotic sequence generation, MD5 key shuffling, and Gilbert 2D curve generation

use md5::{Digest, Md5};

/// Generate Logistic chaotic sequence, returns sorted position indices
pub fn generate_logistic_positions(x1: f64, n: usize) -> Vec<i32> {
    let mut arr: Vec<(f64, usize)> = Vec::with_capacity(n);
    let mut x = x1;
    arr.push((x, 0));

    for i in 1..n {
        x = 3.9999999 * x * (1.0 - x);
        arr.push((x, i));
    }

    // Sort by value
    arr.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Return original position indices
    arr.iter().map(|item| item.1 as i32).collect()
}

/// Generate shuffled sequence using MD5 key (Fisher-Yates shuffle)
pub fn shuffle_with_key(length: usize, key: &str) -> Vec<i32> {
    let mut arr: Vec<i32> = (0..length as i32).collect();

    for i in (1..length).rev() {
        let hash_input = format!("{}{}", key, i);
        let mut hasher = Md5::new();
        hasher.update(hash_input.as_bytes());
        let result = hasher.finalize();

        // Take the first 7 hex digits (28 bits)
        let hex_val = u32::from_be_bytes([0, result[0], result[1], result[2]]) >> 4;
        let rand = (hex_val as usize) % (i + 1);

        arr.swap(rand, i);
    }

    arr
}

/// Generate position sequence for Gilbert 2D space-filling curve
pub fn gilbert2d(width: i32, height: i32) -> Vec<i32> {
    let pixel_count = (width * height) as usize;
    let mut positions = vec![0i32; pixel_count];
    let mut pos = 0usize;

    fn generate2d(
        positions: &mut Vec<i32>,
        pos: &mut usize,
        width: i32,
        height: i32,
        mut x: i32,
        mut y: i32,
        ax: i32,
        ay: i32,
        bx: i32,
        by: i32,
    ) {
        let w = (ax + ay).abs();
        let h = (bx + by).abs();
        let dax = ax.signum();
        let day = ay.signum();
        let dbx = bx.signum();
        let dby = by.signum();

        if h == 1 {
            for _ in 0..w {
                if x >= 0 && x < width && y >= 0 && y < height {
                    positions[*pos] = x + y * width;
                }
                *pos += 1;
                x += dax;
                y += day;
            }
            return;
        }

        if w == 1 {
            for _ in 0..h {
                if x >= 0 && x < width && y >= 0 && y < height {
                    positions[*pos] = x + y * width;
                }
                *pos += 1;
                x += dbx;
                y += dby;
            }
            return;
        }

        let mut ax2 = ax / 2;
        let mut ay2 = ay / 2;
        let mut bx2 = bx / 2;
        let mut by2 = by / 2;
        let w2 = (ax2 + ay2).abs();
        let h2 = (bx2 + by2).abs();

        if 2 * w > 3 * h {
            if (w2 & 1) == 1 && w > 2 {
                ax2 += dax;
                ay2 += day;
            }
            generate2d(positions, pos, width, height, x, y, ax2, ay2, bx, by);
            generate2d(
                positions,
                pos,
                width,
                height,
                x + ax2,
                y + ay2,
                ax - ax2,
                ay - ay2,
                bx,
                by,
            );
        } else {
            if (h2 & 1) == 1 && h > 2 {
                bx2 += dbx;
                by2 += dby;
            }
            generate2d(positions, pos, width, height, x, y, bx2, by2, ax2, ay2);
            generate2d(
                positions,
                pos,
                width,
                height,
                x + bx2,
                y + by2,
                ax,
                ay,
                bx - bx2,
                by - by2,
            );
            generate2d(
                positions,
                pos,
                width,
                height,
                x + (ax - dax) + (bx2 - dbx),
                y + (ay - day) + (by2 - dby),
                -bx2,
                -by2,
                -(ax - ax2),
                -(ay - ay2),
            );
        }
    }

    if width >= height {
        generate2d(
            &mut positions,
            &mut pos,
            width,
            height,
            0,
            0,
            width,
            0,
            0,
            height,
        );
    } else {
        generate2d(
            &mut positions,
            &mut pos,
            width,
            height,
            0,
            0,
            0,
            height,
            width,
            0,
        );
    }

    positions
}
