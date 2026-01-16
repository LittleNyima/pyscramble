//! Scramble algorithms module
//!
//! This module contains all pixel scramble/descramble algorithms:
//! - Tomato Scramble: Based on Gilbert 2D space-filling curve
//! - Per Pixel MD5 Scramble: MD5 hash-based per-pixel scrambling
//! - Row Pixel MD5 Scramble: MD5 hash-based row pixel scrambling
//! - Block MD5 Scramble: MD5 hash-based block scrambling
//! - Pic Row Logistic Scramble: Logistic map-based row scrambling
//! - Pic Row Column Logistic Scramble: Logistic map-based row-column scrambling

pub mod block_md5;
pub mod per_pixel_md5;
pub mod row_column_logistic;
pub mod row_logistic;
pub mod row_md5;
pub mod tomato;

// Re-export all encryption/decryption functions
pub use block_md5::{block_md5_decrypt, block_md5_encrypt};
pub use per_pixel_md5::{per_pixel_md5_decrypt, per_pixel_md5_encrypt};
pub use row_column_logistic::{row_column_logistic_decrypt, row_column_logistic_encrypt};
pub use row_logistic::{row_logistic_decrypt, row_logistic_encrypt};
pub use row_md5::{row_md5_decrypt, row_md5_encrypt};
pub use tomato::{tomato_scramble_decrypt, tomato_scramble_encrypt};
