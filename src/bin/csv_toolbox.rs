// 引入必要的模組
use csv_converter::CsvConverter;
use cargo_tutorial::create_sample_csv_file;

fn main() -> std::io::Result<()> {
    println!("🚀 CSV 工具箱");
    println!("==============");

    // 生成測試 CSV 檔案
    create_sample_csv_file("demo.csv", 50)?;

    // 將 CSV 轉換為 JSON 並儲存
    CsvConverter::convert_csv_to_json_file("demo.csv", "demo.json")?;

    println!("CSV 轉換完成！");
    Ok(())
}
