use lazy_static::lazy_static;
use std::collections::HashMap;
use std::sync::Mutex;

// Global color cache for consistent color assignment across the run
lazy_static! {
    static ref COLOR_CACHE: Mutex<HashMap<i32, opencv::core::Scalar>> = Mutex::new(HashMap::new());
}

/// Get color for a class ID using random color generation with memorization
/// Each class gets a unique random color that persists throughout the run
pub fn get_class_color(class_id: i32) -> opencv::core::Scalar {
    let mut cache = COLOR_CACHE.lock().unwrap();

    // Check if we already assigned a color to this class
    if let Some(&color) = cache.get(&class_id) {
        return color;
    }

    // Generate a new random color for this class
    let new_color = generate_random_color();
    cache.insert(class_id, new_color);
    new_color
}

/// Generate a random bright color using HSV color space
pub fn generate_random_color() -> opencv::core::Scalar {
    use rand::Rng;

    let mut rng = rand::thread_rng();

    // Generate random HSV values
    let hue = rng.gen_range(0.0..360.0);        // Full hue range
    let saturation = rng.gen_range(0.7..1.0);   // High saturation for vivid colors
    let value = rng.gen_range(0.8..1.0);        // High brightness for visibility

    // Convert HSV to RGB
    hsv_to_rgb(hue, saturation, value)
}

/// Convert HSV to RGB color space
pub fn hsv_to_rgb(hue: f64, saturation: f64, value: f64) -> opencv::core::Scalar {
    let c = value * saturation;
    let h = hue / 60.0;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    let m = value - c;

    let (r, g, b) = match h.floor() as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        5 => (c, 0.0, x),
        _ => (0.0, 0.0, 0.0),
    };

    // Convert to 0-255 range and create OpenCV Scalar (BGR order)
    opencv::core::Scalar::new(
        ((b + m) * 255.0) as f64,  // Blue
        ((g + m) * 255.0) as f64,  // Green
        ((r + m) * 255.0) as f64,  // Red
        0.0,                       // Alpha
    )
}
