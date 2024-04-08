pub mod device;
pub mod storage;

#[derive(thiserror::Error, Debug)]
pub enum WebGPUError {
    #[error("WebGPU error: {0}")]
    Message(String),
}

impl From<String> for WebGPUError {
    fn from(e: String) -> Self {
        WebGPUError::Message(e)
    }
}
