use csv::Reader;
use std::fs::File;
use std::collections::HashMap;

/// CSV 轉換器
pub struct CsvConverter;

impl CsvConverter {
    /// 將 CSV 檔案轉換並儲存為 JSON 檔案
    pub fn convert_csv_to_json_file(csv_path: &str, json_path: &str) -> std::io::Result<()> {
        // 開啟 CSV 檔案
        let file = File::open(csv_path)?;
        let mut reader = Reader::from_reader(file);

        // 讀取標題行
        let headers: Vec<String> = reader.headers()?
            .iter()
            .map(|h| h.to_string())
            .collect();

        let mut records = Vec::new();

        // 處理每一行資料
        for result in reader.records() {
            let record = result?;
            let mut row_map = HashMap::new();

            // 將每一行轉換為鍵值對
            for (i, field) in record.iter().enumerate() {
                if i < headers.len() {
                    let header = &headers[i];
                    let value = if let Ok(num) = field.parse::<i64>() {
                        serde_json::Value::Number(num.into())
                    } else if let Ok(num) = field.parse::<f64>() {
                        if let Some(n) = serde_json::Number::from_f64(num) {
                            serde_json::Value::Number(n)
                        } else {
                            serde_json::Value::String(field.to_string())
                        }
                    } else if field.to_lowercase() == "true" {
                        serde_json::Value::Bool(true)
                    } else if field.to_lowercase() == "false" {
                        serde_json::Value::Bool(false)
                    } else {
                        serde_json::Value::String(field.to_string())
                    };

                    row_map.insert(header.clone(), value);
                }
            }

            records.push(row_map);
        }

        // 將記錄序列化為 JSON
        let json_string = serde_json::to_string_pretty(&records)?;

        // 寫入 JSON 檔案
        std::fs::write(json_path, &json_string)?;

        Ok(())
    }
}

