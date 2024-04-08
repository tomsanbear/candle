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

trait Block {
    fn wait(self) -> <Self as futures::Future>::Output
    where
        Self: Sized,
        Self: futures::Future,
    {
        futures::executor::block_on(self)
    }
}

impl<F, T> Block for F where F: futures::Future<Output = T> {}
