use crate::UltralyticsYOLO;
use crate::config;

/// 模型驗證函數，用於驗證 ONNX 模型載入和配置 (Model validation function to verify ONNX model loading and configuration)
pub fn validate_onnx_model(config: &config::Config) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 驗證 ONNX 模型載入 (Validating ONNX Model Loading)");
    println!("========================");

    // 載入 YOLO 模型 (Load the YOLO model)
    println!("📦 載入模型 (Loading model): {}", config.model);
    let model = UltralyticsYOLO::new(config.clone())?;
    println!("✅ 模型載入成功！ (Model loaded successfully!)");

    // 顯示模型資訊 (Display model information)
    println!("\n📊 模型資訊 (Model Information):");
    println!("   任務類型 (Task Type): {:?}", model.task);
    println!("   輸入尺寸 (Input Dimensions): {}x{}", model.width, model.height);
    println!("   批次大小 (Batch Size): {}", model.batch);
    println!("   類別數量 (Number of Classes - nc): {}", model.nc);
    println!("   關鍵點數量 (Number of Keypoints - nk): {}", model.nk);
    println!("   遮罩數量 (Number of Masks - nm): {}", model.nm);
    println!("   信心度閾值 (Confidence Threshold): {:.2}", model.conf);
    println!("   IoU 閾值 (IoU Threshold): {:.2}", model.iou);
    println!("   關鍵點信心度閾值 (Keypoint Confidence Threshold): {:.2}", model.kconf);

    // 顯示執行提供者 (Display execution provider)
    println!("\n⚡ 執行提供者 (Execution Provider):");
    println!("   提供者 (Provider): {:?}", model.engine.ep());

    // 顯示類別名稱（如果可用） (Display class names if available)
    if !model.names.is_empty() {
        println!("\n🏷️  類別名稱 (Class Names - {} classes):", model.names.len());
        for (i, name) in model.names.iter().enumerate() {
            if i < 10 { // 顯示前 10 個類別 (Show first 10 classes)
                println!("   {:2}: {}", i, name);
            } else if i == 10 {
                println!("   ... 還有 {} 個類別 (... and {} more classes)", model.names.len() - 10, model.names.len() - 10);
                break;
            }
        }
    } else {
        println!("\n🏷️  類別名稱 (Class Names): 模型元資料中無法取得 (Not available in model metadata)");
    }


    // 驗證模型後端資訊 (Verify model backend information)
    println!("\n🔧 後端配置 (Backend Configuration):");
    println!("   動態批次 (Dynamic Batch): {}", model.engine.is_batch_dynamic());
    println!("   動態高度 (Dynamic Height): {}", model.engine.is_height_dynamic());
    println!("   動態寬度 (Dynamic Width): {}", model.engine.is_width_dynamic());
    println!("   資料類型 (Data Type): {:?}", model.engine.dtype());

    Ok(())
}