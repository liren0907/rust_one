use ort_tutorial::config;
use ort_tutorial::processor;
use ort_tutorial::validator::validate_onnx_model;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("ORT Tutorial - Simple YOLO Video Processing Demonstration");
    println!("=======================================================");

    // Direct configuration - now using the new constructor with defaults
    let config = config::Config::new("model_weight/yolo11s.onnx");

    println!("Configuration set directly in code");
    println!("Model: {}", config.model);
    println!("Video dimensions: {}x{}", config.width, config.height);
    println!("CUDA enabled: {}", config.cuda);
    println!();

    // First, validate the ONNX model to ensure it loads correctly
    println!("Step 1: Validating ONNX model...");
    validate_onnx_model(&config)?;
    println!("✅ Model validation completed successfully!\n");

    // Then proceed with video processing
    println!("Step 2: Starting video processing...");
    processor::video_processor(&config, "data/test_short.mp4", None)
}
