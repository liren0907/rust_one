use opencv::{
    core::{self, Mat, Scalar},
    highgui,
    imgcodecs,
    imgproc,
    prelude::*,
    videoio,
};
use std::path::Path;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 OpenCV Image Processing Tutorials");
    println!("=====================================\n");

    canny_edge_detection_tutorial()?;
    image_binarization_tutorial()?;
    bounding_box_text_tutorial()?;
    video_annotation_tutorial()?;

    println!("\n✅ OpenCV tutorials completed successfully!");
    Ok(())
}

fn canny_edge_detection_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    let input_path = "data/test.jpg";
    let img = imgcodecs::imread(input_path, imgcodecs::IMREAD_COLOR)?;

    if img.empty() {
        return Err("Failed to read image".into());
    }

    let mut gray = Mat::default();
    imgproc::cvt_color(
        &img,
        &mut gray,
        imgproc::COLOR_BGR2GRAY,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut blurred = Mat::default();
    imgproc::gaussian_blur(
        &gray,
        &mut blurred,
        core::Size::new(5, 5),
        0.0,
        0.0,
        core::BORDER_DEFAULT,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut edges = Mat::default();
    imgproc::canny(&blurred, &mut edges, 50.0, 150.0, 3, false)?;

    let params = core::Vector::<i32>::new();
    let output_path = "data/test_edges.jpg";
    imgcodecs::imwrite(output_path, &edges, &params)?;

    Ok(())
}

fn image_binarization_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    let input_path = "data/test.jpg";
    let img = imgcodecs::imread(input_path, imgcodecs::IMREAD_COLOR)?;

    if img.empty() {
        return Err("Failed to read image for binarization".into());
    }

    let mut gray = Mat::default();
    imgproc::cvt_color(
        &img,
        &mut gray,
        imgproc::COLOR_BGR2GRAY,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut binary = Mat::default();
    let threshold_value = 127.0;
    let max_value = 255.0;

    imgproc::threshold(
        &gray,
        &mut binary,
        threshold_value,
        max_value,
        imgproc::THRESH_BINARY,
    )?;

    let binary_output = "data/test_binary.jpg";
    let params = core::Vector::<i32>::new();
    imgcodecs::imwrite(binary_output, &binary, &params)?;

    Ok(())
}

fn bounding_box_text_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    let input_path = "data/test.jpg";
    let img = imgcodecs::imread(input_path, imgcodecs::IMREAD_COLOR)?;

    if img.empty() {
        return Err("Failed to read image for annotation".into());
    }

    let mut annotated_img = img.clone();

    let bbox = core::Rect::new(100, 100, 200, 150);
    imgproc::rectangle(
        &mut annotated_img,
        bbox,
        Scalar::new(0.0, 0.0, 255.0, 0.0),
        3,
        imgproc::LINE_8,
        0,
    )?;

    let text = "Object";
    imgproc::put_text(
        &mut annotated_img,
        text,
        core::Point::new(bbox.x, bbox.y - 10),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.8,
        Scalar::new(0.0, 0.0, 255.0, 0.0),
        2,
        imgproc::LINE_8,
        false,
    )?;

    let annotated_output = "data/test_annotated.jpg";
    let params = core::Vector::<i32>::new();
    imgcodecs::imwrite(annotated_output, &annotated_img, &params)?;

    Ok(())
}

fn video_annotation_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    let video_path = "data/test_short.mp4";

    if !Path::new(video_path).exists() {
        return Ok(());
    }

    let mut cap = videoio::VideoCapture::from_file(video_path, videoio::CAP_ANY)?;
    if !cap.is_opened()? {
        return Err("Failed to open video file".into());
    }

    let fps = cap.get(videoio::CAP_PROP_FPS)?;
    let width = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let height = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;

    let window_name = "Video Annotation Tutorial";
    highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE)?;

    let output_video_path = "data/annotated_video.mp4";
    let fourcc = videoio::VideoWriter::fourcc('m', 'p', '4', 'v')?;
    let mut writer = videoio::VideoWriter::new(
        output_video_path,
        fourcc,
        fps,
        core::Size::new(width, height),
        true,
    )?;

    if !writer.is_opened()? {
        return Err("Failed to create output video file".into());
    }

    let mut frame = Mat::default();

    while cap.read(&mut frame)? {
        if frame.empty() {
            break;
        }

        let mut annotated_frame = frame.clone();

        let box_width = 200;
        let box_height = 100;
        let center_x = width / 2;
        let center_y = height / 2;
        let box_x = center_x - box_width / 2;
        let box_y = center_y - box_height / 2;

        let bbox = core::Rect::new(box_x, box_y, box_width, box_height);
        imgproc::rectangle(
            &mut annotated_frame,
            bbox,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            3,
            imgproc::LINE_8,
            0,
        )?;

        let text = "Center Detection";
        let text_x = box_x;
        let text_y = box_y - 10;
        imgproc::put_text(
            &mut annotated_frame,
            text,
            core::Point::new(text_x, text_y),
            imgproc::FONT_HERSHEY_DUPLEX,
            0.8,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            false,
        )?;

        writer.write(&annotated_frame)?;

        highgui::imshow(window_name, &annotated_frame)?;

        let key = highgui::wait_key(30)?;
        match key {
            113 | 27 => {
                break;
            }
            _ => {}
        }
    }

    cap.release()?;
    writer.release()?;
    highgui::destroy_window(window_name)?;

    Ok(())
}
