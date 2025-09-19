//! 測試資料產生器模組 - 用於產生測試資料檔案的簡單函式

use std::fs::File;
use csv::Writer;

/// 產生包含測試使用者資料的範例 CSV 檔案
pub fn create_sample_csv_file(filename: &str, record_count: usize) -> std::io::Result<()> {
    let file = File::create(filename)?;
    let mut writer = Writer::from_writer(file);

    // 寫入 CSV 標題列
    writer.write_record(&["id", "name", "age", "city", "value", "active"])?;

    let cities = ["New York", "London", "Tokyo", "Paris"];

    // 產生測試記錄（為簡單起見使用確定性生成）
    for i in 1..=record_count {
        let name = format!("User{}", i);
        let age = 20 + (i % 50); // 年齡介於 20-69 歲之間
        let city = cities[i % cities.len()];
        let value = 100.0 + (i as f64 * 10.0);
        let active = (i % 3) != 0; // 每第 3 筆記錄設為非活躍狀態

        writer.write_record(&[
            &i.to_string(),
            &name,
            &age.to_string(),
            city,
            &format!("{:.2}", value),
            &active.to_string()
        ])?;
    }

    writer.flush()?;
    Ok(())
}

