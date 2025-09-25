use crate::UltralyticsYOLO;
use crate::YOLOResult;
use crate::config;
use opencv::Error;
use opencv::core::Mat;

/// Draw annotations (bounding boxes and labels) on a frame based on YOLO predictions
pub fn annotation_drawer(
    frame: &Mat,
    predictions: &[YOLOResult],
    model: &UltralyticsYOLO,
    _config: &config::Config,
) -> Result<Mat, Error> {
    // Create a copy of the frame to draw on (avoid modifying original)
    let mut annotated_frame = frame.clone();

    // Process predictions and draw annotations
    if let Some(prediction) = predictions.get(0) {
        if let Some(bboxes) = &prediction.bboxes {
            for bbox in bboxes {
                // Get color for the class - using a simple color generation function
                let class_id = bbox.id() as i32;
                let hue = (class_id * 137) % 360; // Generate different hues for different classes
                let color = opencv::core::Scalar::new(
                    (hue as f64 * 255.0 / 360.0) % 255.0,
                    200.0,
                    200.0,
                    0.0,
                );

                // Draw bounding box
                let pt1 = opencv::core::Point::new(bbox.xmin() as i32, bbox.ymin() as i32);
                let pt2 = opencv::core::Point::new(
                    (bbox.xmin() + bbox.width()) as i32,
                    (bbox.ymin() + bbox.height()) as i32,
                );

                opencv::imgproc::rectangle(
                    &mut annotated_frame,
                    opencv::core::Rect::from_points(pt1, pt2),
                    color,
                    2,
                    opencv::imgproc::LINE_8,
                    0,
                )?;

                // Draw label
                let class_name = model
                    .names
                    .get(bbox.id() as usize)
                    .map_or("unknown", |s| s.as_str());

                let label = format!("{} {:.2}", class_name, bbox.confidence());
                let font_scale = 0.5;
                let thickness = 1;
                let font_face = opencv::imgproc::FONT_HERSHEY_SIMPLEX;

                // Get text size
                let text_size = opencv::imgproc::get_text_size(
                    &label, font_face, font_scale, thickness, &mut 0,
                )?;

                // Draw background rectangle for text
                opencv::imgproc::rectangle(
                    &mut annotated_frame,
                    opencv::core::Rect::new(
                        pt1.x,
                        pt1.y - text_size.height - 5,
                        text_size.width + 10,
                        text_size.height + 10,
                    ),
                    color,
                    -1, // filled
                    opencv::imgproc::LINE_8,
                    0,
                )?;

                // Draw text
                opencv::imgproc::put_text(
                    &mut annotated_frame,
                    &label,
                    opencv::core::Point::new(pt1.x + 5, pt1.y - 5),
                    font_face,
                    font_scale,
                    opencv::core::Scalar::new(255.0, 255.0, 255.0, 0.0),
                    thickness,
                    opencv::imgproc::LINE_8,
                    false,
                )?;
            }
        }
    }

    Ok(annotated_frame)
}
