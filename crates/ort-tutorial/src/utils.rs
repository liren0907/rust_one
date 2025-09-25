use chrono::{DateTime, Utc};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

pub fn generate_output_path(video_path: &str) -> PathBuf {
    let input_path = Path::new(video_path);
    let filename = input_path
        .file_stem()
        .unwrap_or_else(|| OsStr::new("video"))
        .to_string_lossy();

    let now: DateTime<Utc> = Utc::now();
    let timestamp = now.format("%Y%m%d_%H%M%S").to_string();

    let output_filename = format!("{}_output_{}.mp4", filename, timestamp);

    Path::new("output").join(output_filename)
}
