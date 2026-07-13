use crate::source::{
    AFFINE, BINARY, CAST, CONV, FILL, GATED_DELTA, GATED_DELTA_CHUNK, GATED_DELTA_V2, GEMV,
    INDEXING, MLX_GEMM, MLX_SORT, QUANTIZED, RANDOM, REDUCE, SDPA, SKINNY_GEMM, SORT, TERNARY,
    UNARY,
};
use crate::utils::get_env_bool;
use crate::{
    ComputePipeline, ConstantValues, Device, Function, Library, MTLCompileOptions,
    MTLMathFloatingPointFunctions, MTLMathMode, MetalKernelError, Source,
};
use objc2::available;
use objc2::rc::Retained;
use std::collections::HashMap;
use std::sync::RwLock;

#[derive(Debug, Clone)]
pub enum KernelName {
    Ref(&'static str),
    Value(String),
}

impl AsRef<str> for KernelName {
    fn as_ref(&self) -> &str {
        match self {
            Self::Ref(r) => r,
            Self::Value(v) => v.as_str(),
        }
    }
}

impl std::hash::Hash for KernelName {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Ref(r) => r.hash(state),
            Self::Value(v) => v.hash(state),
        }
    }
}

impl PartialEq for KernelName {
    fn eq(&self, other: &Self) -> bool {
        let v1: &str = self.as_ref();
        let v2: &str = other.as_ref();
        v1 == v2
    }
}

impl Eq for KernelName {}

impl From<&'static str> for KernelName {
    fn from(value: &'static str) -> Self {
        Self::Ref(value)
    }
}

impl From<String> for KernelName {
    fn from(value: String) -> Self {
        Self::Value(value)
    }
}

/// Multiply-xor hasher (the FxHash construction) for the kernel caches: the
/// pipeline map is probed once per dispatch with a short kernel-name key, and
/// SipHash's per-probe setup dominates that hit path. DoS resistance is
/// irrelevant here — keys are compile-time kernel names, never external input.
#[derive(Default)]
struct FxHasher {
    hash: u64,
}

impl std::hash::Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
        for chunk in bytes.chunks(8) {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            self.hash = (self.hash.rotate_left(5) ^ u64::from_le_bytes(buf)).wrapping_mul(SEED);
        }
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

type FxBuildHasher = std::hash::BuildHasherDefault<FxHasher>;
type Libraries = HashMap<Source, Library, FxBuildHasher>;
type Pipelines = HashMap<(KernelName, Option<ConstantValues>), ComputePipeline, FxBuildHasher>;

#[derive(Debug)]
pub struct Kernels {
    libraries: RwLock<Libraries>,
    pipelines: RwLock<Pipelines>,
}

impl Default for Kernels {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernels {
    pub fn new() -> Self {
        let libraries = RwLock::new(Libraries::default());
        let pipelines = RwLock::new(Pipelines::default());
        Self {
            libraries,
            pipelines,
        }
    }

    fn get_library_source(&self, source: Source) -> &'static str {
        match source {
            Source::Affine => AFFINE,
            Source::Binary => BINARY,
            Source::Cast => CAST,
            Source::Conv => CONV,
            Source::Fill => FILL,
            Source::Gemm => MLX_GEMM,
            Source::Gemv => GEMV,
            Source::Indexing => INDEXING,
            Source::MlxSort => MLX_SORT,
            Source::Quantized => QUANTIZED,
            Source::Random => RANDOM,
            Source::Reduce => REDUCE,
            Source::Sort => SORT,
            Source::Ternary => TERNARY,
            Source::Unary => UNARY,
            Source::Sdpa => SDPA,
            Source::GatedDelta => GATED_DELTA,
            Source::GatedDeltaChunk => GATED_DELTA_CHUNK,
            Source::GatedDeltaV2 => GATED_DELTA_V2,
            Source::SkinnyGemm => SKINNY_GEMM,
            Source::Dspark => DSPARK,
        }
    }

    /// Load the give library from its [`source`].
    /// If this has been previously loaded it will just fetch it from cache.
    pub fn load_library(
        &self,
        device: &Device,
        source: Source,
    ) -> Result<Library, MetalKernelError> {
        // Fast path: cache hits (every call after the first per source) take
        // only a shared read lock.
        if let Some(lib) = self.libraries.read()?.get(&source) {
            return Ok(lib.clone());
        }
        let mut libraries = self.libraries.write()?;
        // Re-probe under the write lock: another thread may have compiled
        // this source between the read and write acquisitions.
        if let Some(lib) = libraries.get(&source) {
            Ok(lib.clone())
        } else {
            let lib = {
                let source_content = self.get_library_source(source);
                let compile_options = get_compile_options();
                device
                    .new_library_with_source(source_content, Some(&compile_options))
                    .map_err(|e| MetalKernelError::LoadLibraryError(e.to_string()))?
            };
            libraries.insert(source, lib.clone());
            Ok(lib)
        }
    }

    fn load_function(
        &self,
        device: &Device,
        source: Source,
        name: &str,
        constants: Option<&ConstantValues>,
    ) -> Result<Function, MetalKernelError> {
        let func = self
            .load_library(device, source)?
            .get_function(name, constants)?;
        Ok(func)
    }

    /// Load the give pipeline
    /// loads the library from source, then gets the function [`name`] from
    /// that source
    pub fn load_pipeline_with_constants(
        &self,
        device: &Device,
        source: Source,
        name: impl Into<KernelName>,
        constants: Option<ConstantValues>,
    ) -> Result<ComputePipeline, MetalKernelError> {
        let key = (name.into(), constants);
        // Fast path: cache hits (the per-dispatch steady state) take only a
        // shared read lock.
        if let Some(pipeline) = self.pipelines.read()?.get(&key) {
            return Ok(pipeline.clone());
        }
        let mut pipelines = self.pipelines.write()?;
        // Re-probe under the write lock: another thread may have built this
        // pipeline between the read and write acquisitions.
        if let Some(pipeline) = pipelines.get(&key) {
            Ok(pipeline.clone())
        } else {
            let (name, constants) = key;
            let func = self.load_function(device, source, name.as_ref(), constants.as_ref())?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| MetalKernelError::FailedToCreatePipeline(e.to_string()))?;
            pipelines.insert((name, constants), pipeline.clone());

            Ok(pipeline)
        }
    }

    /// Load the give pipeline
    /// loads the library from source, then gets the function [`name`] from
    /// that source (without constants)
    pub fn load_pipeline(
        &self,
        device: &Device,
        source: Source,
        name: impl Into<KernelName>,
    ) -> Result<ComputePipeline, MetalKernelError> {
        self.load_pipeline_with_constants(device, source, name, None)
    }
}

fn get_compile_options() -> Retained<MTLCompileOptions> {
    let compile_options = MTLCompileOptions::new();
    //unsafe { compile_options.setEnableLogging(true) };

    let fast_math_enabled = get_env_bool("CANDLE_METAL_ENABLE_FAST_MATH", true);
    // Ref availability:
    // https://developer.apple.com/documentation/metal/mtlcompileoptions/mathmode
    if available!(macos = 15, ios = 18) {
        if fast_math_enabled {
            compile_options.setMathMode(MTLMathMode::Fast);
            compile_options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Fast);
        } else {
            compile_options.setMathMode(MTLMathMode::Relaxed);
            compile_options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        }
    } else {
        // For older OS versions we use the old api
        #[allow(deprecated)]
        compile_options.setFastMathEnabled(fast_math_enabled);
    }
    compile_options
}
