use candle_webgpu_kernels::WebGPUKernelError;

pub mod device;
pub mod storage;

#[derive(thiserror::Error, Debug)]
pub enum WebGPUError {
    #[error("WebGPU error: {0}")]
    Message(String),
    #[error("WebGPUKernel error: {0}")]
    Kernel(WebGPUKernelError),
}

impl From<String> for WebGPUError {
    fn from(e: String) -> Self {
        WebGPUError::Message(e)
    }
}

impl From<WebGPUKernelError> for WebGPUError {
    fn from(e: WebGPUKernelError) -> Self {
        WebGPUError::Kernel(e)
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
