use crate::{BlitCommandEncoder, ComputeCommandEncoder};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus};
use std::borrow::Cow;
use std::ptr::NonNull;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

#[derive(Clone, Debug, PartialEq)]
pub enum CommandStatus {
    Available,
    Encoding,
    Done,
}

#[derive(Debug)]
pub struct CommandSemaphore {
    pub cond: Condvar,
    pub status: Mutex<CommandStatus>,
}

impl CommandSemaphore {
    pub fn new() -> CommandSemaphore {
        CommandSemaphore {
            cond: Condvar::new(),
            status: Mutex::new(CommandStatus::Available),
        }
    }

    pub fn wait_until<F: FnMut(&mut CommandStatus) -> bool>(
        &self,
        mut f: F,
    ) -> MutexGuard<'_, CommandStatus> {
        self.cond
            .wait_while(self.status.lock().unwrap(), |s| !f(s))
            .unwrap()
    }

    pub fn set_status(&self, status: CommandStatus) {
        *self.status.lock().unwrap() = status;
        // We notify the condvar that the value has changed.
        self.cond.notify_one();
    }

    pub fn when<T, B: FnMut(&mut CommandStatus) -> bool, F: FnMut() -> T>(
        &self,
        b: B,
        mut f: F,
        next: Option<CommandStatus>,
    ) -> T {
        let mut guard = self.wait_until(b);
        let v = f();
        if let Some(status) = next {
            *guard = status;
            self.cond.notify_one();
        }
        v
    }
}

#[derive(Clone, Debug)]
pub struct CommandBuffer {
    raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    semaphore: Arc<CommandSemaphore>,
}

unsafe impl Send for CommandBuffer {}
unsafe impl Sync for CommandBuffer {}

impl CommandBuffer {
    pub fn new(
        raw: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        semaphore: Arc<CommandSemaphore>,
    ) -> Self {
        Self { raw, semaphore }
    }

    pub fn compute_command_encoder(&self) -> ComputeCommandEncoder {
        self.as_ref()
            .computeCommandEncoder()
            .map(|raw| {
                ComputeCommandEncoder::new_with_buffer(
                    raw,
                    Arc::clone(&self.semaphore),
                    self.raw.clone(),
                )
            })
            .unwrap()
    }

    pub fn blit_command_encoder(&self) -> BlitCommandEncoder {
        self.as_ref()
            .blitCommandEncoder()
            .map(|raw| BlitCommandEncoder::new(raw, Arc::clone(&self.semaphore)))
            .unwrap()
    }

    pub fn commit(&self) {
        self.raw.commit()
    }

    pub fn enqueue(&self) {
        self.raw.enqueue()
    }

    pub fn set_label(&self, label: &str) {
        self.as_ref().setLabel(Some(&NSString::from_str(label)))
    }

    /// Read back the buffer's label (as set by `set_label`). Used by
    /// programmatic profilers to attribute timings to kernels.
    pub fn label(&self) -> Option<String> {
        unsafe {
            self.raw.label().map(|ns| {
                let c_str = core::ffi::CStr::from_ptr(ns.UTF8String());
                c_str.to_string_lossy().into_owned()
            })
        }
    }

    pub fn status(&self) -> MTLCommandBufferStatus {
        self.raw.status()
    }

    pub fn error(&self) -> Option<Cow<'_, str>> {
        unsafe {
            self.raw.error().map(|error| {
                let description = error.localizedDescription();
                let c_str = core::ffi::CStr::from_ptr(description.UTF8String());
                c_str.to_string_lossy()
            })
        }
    }

    pub fn wait_until_completed(&self) {
        self.raw.waitUntilCompleted();
    }

    /// GPU-side kernel start time in seconds (CFTimeInterval epoch), valid
    /// only after the buffer has completed. Pair with `kernel_end_time()`
    /// for per-buffer GPU duration — the primary signal for our
    /// `metal-profile`-feature profiler in bitnet-rs.
    pub fn kernel_start_time(&self) -> f64 {
        self.raw.kernelStartTime()
    }

    /// GPU-side kernel end time in seconds (CFTimeInterval epoch).
    pub fn kernel_end_time(&self) -> f64 {
        self.raw.kernelEndTime()
    }

    /// Register a completion callback that fires on a Metal-internal thread
    /// as soon as the buffer finishes executing. `handler` receives this
    /// `CommandBuffer` (cloned) so it can read `kernel_start_time` /
    /// `kernel_end_time` without blocking the producer. Used by programmatic
    /// GPU profilers (e.g. the `metal-profile` feature in bitnet-rs).
    ///
    /// Safety: the callback runs on a different thread than the caller.
    /// The `Fn` closure must be `Send + Sync + 'static`.
    pub fn add_completed_handler<F>(&self, handler: F)
    where
        F: Fn(&CommandBuffer) + Send + Sync + 'static,
    {
        // Clone a lightweight handle to this CommandBuffer so the closure
        // keeps the underlying MTLCommandBuffer alive and can query timings.
        let cb_copy = self.clone();
        let block = block2::RcBlock::new(
            move |_buf: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                handler(&cb_copy);
            },
        );
        // SAFETY: `addCompletedHandler:` retains the block; `RcBlock::as_ptr`
        // yields a valid `*mut block2::DynBlock<...>` with the exact signature
        // Metal expects (`^(id<MTLCommandBuffer>)`).
        unsafe {
            let ptr = block2::RcBlock::<
                dyn Fn(NonNull<ProtocolObject<dyn MTLCommandBuffer>>),
            >::as_ptr(&block);
            self.raw.addCompletedHandler(ptr as _);
        }
        // `block` drops here, but Metal has retained the block internally;
        // RcBlock semantics match ObjC's `copy` requirement for captured
        // completion handlers.
    }
}

impl AsRef<ProtocolObject<dyn MTLCommandBuffer>> for CommandBuffer {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLCommandBuffer> {
        &self.raw
    }
}
