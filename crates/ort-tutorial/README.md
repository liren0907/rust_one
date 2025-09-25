# ORT Tutorial: UltralyticsYOLO based ONNX runtime inference

## Project Highlights

### Core Features
- **YOLO Tasks**: Object detection, pose estimation, segmentation, classification
- **Execution**: CPU, CUDA GPU, TensorRT with automatic provider selection
- **Performance**: Batch processing, dynamic shapes, FP16 optimization

---

## Architecture

### Module Layout

```
src/
├── lib.rs              # Public API
├── main.rs             # Demo application
├── model.rs            # YOLOv8 core logic
├── ort_backend.rs      # ONNX Runtime backend
├── yolo_result.rs      # Result data structures
├── config.rs           # Configuration management
├── nms.rs              # Non-Maximum Suppression
├── validator.rs        # Model validation utilities
├── processor.rs        # Video processing pipeline
├── drawer.rs           # Annotation drawing utilities
├── utils.rs            # Utility functions
└── color.rs            # Visualization color helpers
```


## Environment Setup

### Requirements

- ONNX Runtime: downloaded automatically via the `ort` crate
- OpenCV: for image processing and video I/O


### Build and Run

```bash
# 1) Enter the project root
cd /path/to/your/project

# 2) Build (release)
cargo build --release -p ort-tutorial

# 3) Run the tutorial program
cargo run -p ort-tutorial

# Or use your own runner script
./run_ort_tutorial.sh
```



## Module Overview

### **Usage Examples**

#### Basic Usage
```rust
use ort_tutorial::YOLOv8;
use ort_tutorial::config;

// Build configuration
let config = config::Config {
    model: "model_weight/onnx/version/yolo11s.onnx".to_string(),
    height: 640,
    width: 640,
    conf: 0.3,
    iou: 0.45,
    ..Default::default()
};

// Initialize model
let mut model = YOLOv8::new(config)?;

// Load an image
let img = opencv::imgcodecs::imread("test.jpg", opencv::imgcodecs::IMREAD_COLOR)?;

// Run inference
let results = model.run_mat(&vec![img])?;

// Handle results
if let Some(result) = results.get(0) {
    if let Some(bboxes) = &result.bboxes {
        println!("Detected {} objects", bboxes.len());
        for bbox in bboxes {
            println!("Class: {}, Confidence: {:.2}", bbox.id(), bbox.confidence());
        }
    }
}
```

#### Video Processing
```rust
// Process frames using the high-level video processor
processor::video_processor(&config, "input.mp4", Some("output.mp4"))?;
```

#### Individual Components
```rust
// Run inference on a single frame
let frame = opencv::imgcodecs::imread("test.jpg", opencv::imgcodecs::IMREAD_COLOR)?;
let predictions = model.run_mat(&vec![frame.clone()])?;

// Create annotated frame using the drawing utility
let annotated_frame = ort_tutorial::annotation_drawer(&frame, &predictions, &model, &config)?;

// Generate timestamped output path
let output_path = ort_tutorial::generate_output_path("input_video.mp4");
```

---

## API Reference

### YOLOv8 Struct

```rust
pub struct YOLOv8 {
    pub engine: OrtBackend,        // ONNX Runtime engine
    pub nc: u32,                   // number of classes
    pub nk: u32,                   // number of keypoints
    pub nm: u32,                   // number of masks
    pub height: u32,               // input height
    pub width: u32,                // input width
    pub task: YOLOTask,            // task type
    // ... other fields
}
```

### Main Methods

- `new(config: Config) -> Result<Self>` – create a new model instance
- `run_mat(xs: &Vec<Mat>) -> Result<Vec<YOLOResult>>` – process OpenCV Mat images and return results

### Utility Functions

- `annotation_drawer(frame: &Mat, predictions: &[YOLOResult], model: &YOLOv8, config: &Config) -> Result<Mat>` – draw bounding boxes and labels on a frame
- `generate_output_path(video_path: &str) -> PathBuf` – generate timestamped output path
- `validate_onnx_model(config: &Config) -> Result<()>` – validate ONNX model loading and configuration

### YOLOResult Struct

```rust
pub struct YOLOResult {
    pub probs: Option<Embedding>,            // classification probabilities
    pub bboxes: Option<Vec<Bbox>>,           // bounding boxes
    pub keypoints: Option<Vec<Vec<Point2>>>, // keypoints
    pub masks: Option<Vec<Vec<u8>>>,         // segmentation masks
}
```

### Bounding Box (Bbox)

```rust
pub struct Bbox {
    xmin: f32, ymin: f32,     // top-left coordinates
    width: f32, height: f32,  // dimensions
    id: usize,                // class id
    confidence: f32,          // confidence score
}
```

---

## Configuration Options

### Basic Config

```rust
pub struct Config {
    pub model: String,        // ONNX model path
    pub cuda: bool,           // use CUDA
    pub height: u32,          // input height
    pub width: u32,           // input width
    pub device_id: u32,       // GPU device id
    pub batch: u32,           // batch size
    pub conf: f32,            // confidence threshold (0.0–1.0)
    pub iou: f32,             // IoU threshold (0.0–1.0)
    pub kconf: f32,           // keypoint confidence threshold
    pub window_view: bool,    // show window
    pub profile: bool,        // enable profiling
    // ... other options
}
```

### Execution Providers

| Provider | Description | Typical Use |
|---------|-------------|-------------|
| CPU | CPU execution | Development/testing, constrained environments |
| CUDA | NVIDIA GPU | High-performance inference |
| TensorRT | NVIDIA TensorRT | Optimized production deployments |

---

## Performance Analysis

### Enable Profiling

```rust
let config = config::Config {
    profile: true,  // enable profiling
    // ... other options
};
```

### Example Output

```
[Model Preprocess]: 2.34ms
[Model Inference]: 15.67ms
[Model Postprocess]: 1.23ms
Total processing time: 19.24ms
```

---

## Visualization

### Automatic Color Assignment

The project uses HSV space to generate vivid random colors. Each class gets a unique color that remains consistent across the run.

```rust
use ort_tutorial::color;

// Get a color for a class id
let color = color::get_class_color(class_id);
```

### Supported Drawing Features

- Bounding boxes with unique colors per class
- Class labels with confidence scores
- Background rectangles for better text readability
- Automatic color assignment and caching
- Pure functional drawing utilities

### Architecture Highlights

**✨ Modular Design**
- Clean separation between video processing and annotation drawing
- Pure functions that are easy to test and reuse
- Independent components that can be used standalone

**🚀 Performance Features**
- Efficient OpenCV integration for real-time processing
- Automatic timestamped output path generation
- Optimized memory usage with minimal frame copying
- Configurable processing parameters

---

## Technical Details

### Core Processing Pipeline

1. **Model Inference**: Run YOLO model with ONNX Runtime backend
2. **Result Processing**: Parse outputs and apply confidence filtering
3. **Annotation Drawing**: Generate visual annotations with bounding boxes and labels
4. **Output Generation**: Create timestamped output files automatically

### Architecture Benefits

- **Separation of Concerns**: Each module has a single, clear responsibility
- **Pure Functions**: Drawing utilities are stateless and predictable
- **Composability**: Functions can be chained and reused independently
- **Testability**: Modular design enables focused unit testing
- **Maintainability**: Clean interfaces between components

### Memory Management

- Uses `ndarray` for efficient array operations
- Batch processing reduces allocation overhead
- Efficient conversion between host and device memory (where applicable)
- Pure annotation functions avoid unnecessary frame copying

### Utility Functions

- **Smart Path Generation**: Automatic timestamped output paths
- **Color Management**: Consistent class-based color assignment
- **Validation**: Comprehensive model loading verification
- **Error Handling**: Robust error propagation throughout the pipeline

---

## Advanced Topics

### Custom Models

```rust
// Supports any YOLOv8/11 ONNX model
// Ensure the model contains correct metadata
let config = Config {
    model: "your_custom_model.onnx".to_string(),
    // Automatic task detection and parameters
    ..Default::default()
};
```

### Batch Processing

```rust
// Process multiple images simultaneously
let images = vec![img1, img2, img3, img4];
let batch_size = 4;

let results = model.run_mat(&images)?;
// Handle results for all 4 images
```

### Modular Architecture

The current architecture demonstrates excellent separation of concerns:
- **Independent Components**: Each module can be used standalone
- **Pure Functions**: Drawing utilities are stateless and predictable
- **Composable Design**: Functions can be chained and reused
- **Extensible**: Easy to add new annotation types or processing steps

### Advanced Usage Patterns

```rust
// Use only the components you need
let predictions = model.run_mat(&vec![frame])?;
let annotated_frame = ort_tutorial::annotation_drawer(&frame, &predictions, &model, &config)?;

// Or use the high-level processor for complete video processing
processor::video_processor(&config, "input.mp4", Some("output.mp4"))?;
```

