pub mod ops;

#[derive(thiserror::Error, Debug)]
pub enum WebGPUKernelError {
    #[error("Failed to create pipeline")]
    FailedToCreatePipeline(String),
}

#[cfg(test)]
mod tests;
