#[derive(Debug, Clone)]
pub struct Config {
    pub model: String,
    pub cuda: bool,
    pub height: u32,
    pub width: u32,
    pub device_id: u32,
    pub batch: u32,
    pub batch_min: u32,
    pub batch_max: u32,
    pub profile: bool,
    pub window_view: bool,
    pub trt: bool,
    pub fp16: bool,
    pub nc: Option<u32>,
    pub nk: Option<u32>,
    pub nm: Option<u32>,
    pub conf: f32,
    pub iou: f32,
    pub kconf: f32,
    pub plot: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: String::new(),
            cuda: false,
            height: 640,
            width: 640,
            device_id: 0,
            batch: 1,
            batch_min: 1,
            batch_max: 32,
            profile: false,
            window_view: true,
            trt: false,
            fp16: false,
            nc: None,
            nk: None,
            nm: None,
            conf: 0.3,
            iou: 0.45,
            kconf: 0.55,
            plot: false,
        }
    }
}

impl Config {
    // 建立一個新的 Config 實例，指定模型路徑並為其他欄位使用預設值
    // * `model` - 模型檔案路徑，可接受 &str 或 String 型別
    // 運作原理：
    // 1. 使用 `impl Into<String>` 讓參數可以接受任何能轉換成 String 的型別
    // 2. 透過 `model.into()` 將參數轉換為 String 型別
    // 3. 使用 `..Default::default()` 為其他所有欄位填入預設值`
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }
}
