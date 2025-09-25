use crate::UltralyticsYOLO;
use crate::config;
use crate::generate_output_path;
use crate::annotation_drawer;
use opencv::prelude::*;
use std::path::Path;


/// Simple video processing demonstration using a single YOLO model
pub fn video_processor(config: &config::Config, video_path: &str, output_path: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting simple video processing demonstration");

    // Load the YOLO model
    let mut model = UltralyticsYOLO::new(config.clone())?;

    // Use provided video path
    println!("Processing video: {}", video_path);

    // Open video capture
    let mut cap = opencv::videoio::VideoCapture::from_file(video_path, opencv::videoio::CAP_ANY)?;
    if !cap.is_opened()? {
        return Err(format!("Could not open video file: {}", video_path).into());
    }

    let frame_width = cap.get(opencv::videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let frame_height = cap.get(opencv::videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    let fps = cap.get(opencv::videoio::CAP_PROP_FPS)?;
    println!(
        "Video dimensions: {}x{}, FPS: {}",
        frame_width, frame_height, fps
    );

    // Determine output path - use provided path or generate timestamped path
    let output_path = match output_path {
        Some(path) => Path::new(path).to_path_buf(),
        None => generate_output_path(video_path)
    };
    let fourcc = opencv::videoio::VideoWriter::fourcc('m', 'p', '4', 'v')?;
    let mut writer = opencv::videoio::VideoWriter::new(
        output_path.to_str().unwrap(),
        fourcc,
        fps,
        opencv::core::Size::new(frame_width, frame_height),
        true,
    )?;

    if !writer.is_opened()? {
        println!("Warning: Could not create video writer. Output will not be saved.");
    }

    let mut frame_count = 0;
    let start_time = std::time::Instant::now();

    println!("Processing video frames...");

    let mut frame = opencv::core::Mat::default();
    while cap.read(&mut frame)? {
        if frame.empty() {
            break;
        }
        frame_count += 1;

        // Run inference on the frame
        let predictions = model.run_mat(&vec![frame.clone()])?;

        // Create annotated frame using the annotation_drawer function
        let annotated_frame = annotation_drawer(&frame, &predictions, &model, &config)?;

        // Write frame to output video
        if writer.is_opened()? {
            writer.write(&annotated_frame)?;
        }

        // Display frame if window_view is enabled
        if config.window_view {
            opencv::highgui::imshow("ORT Tutorial - YOLO Detection", &annotated_frame)?;
            let key = opencv::highgui::wait_key(1)?;
            if key == 27 {
                // ESC key
                break;
            }
        }

        // Print progress every 30 frames
        if frame_count % 30 == 0 {
            println!("Processed {} frames", frame_count);
        }
    }

    let elapsed = start_time.elapsed();
    println!("Processing complete!");
    println!("Total frames processed: {}", frame_count);
    println!("Processing time: {:.2}s", elapsed.as_secs_f64());
    println!(
        "Average FPS: {:.2}",
        frame_count as f64 / elapsed.as_secs_f64()
    );

    if writer.is_opened()? {
        println!("Output video saved to: {}", output_path.display());
    }

    Ok(())
}