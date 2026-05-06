use crate::metal::{
    BlitCommandEncoder, CommandBuffer, CommandSemaphore, CommandStatus, ComputeCommandEncoder,
};
use crate::MetalKernelError;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandBufferStatus, MTLCommandQueue};
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "profile")]
use std::sync::RwLock;
use std::sync::{Arc, Mutex};

#[cfg(feature = "profile")]
use crate::metal::profile::{cpu_now_ns, ArgValue, CommandBufferProfile, Lane, MetalProfiler};

// Use Retained when appropriate. Gives us a more elegant way of handling memory (peaks) than autoreleasepool.
// https://docs.rs/objc2/latest/objc2/rc/struct.Retained.html
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

const DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER: usize = 50;
const DEFAULT_CANDLE_METAL_COMMAND_POOL_SIZE: usize = 5;

/// Creates a new command buffer from the queue with an attached semaphore for tracking its state.
pub fn create_command_buffer(
    command_queue: &CommandQueue,
    semaphore: Arc<CommandSemaphore>,
) -> Result<CommandBuffer, MetalKernelError> {
    command_queue
        .commandBuffer()
        .map(|raw| CommandBuffer::new(raw, semaphore))
        .ok_or(MetalKernelError::FailedToCreateResource(
            "CommandBuffer".to_string(),
        ))
}

struct EntryState {
    current: CommandBuffer,
    in_flight: Vec<CommandBuffer>,
}

/// A pool entry containing a command buffer, its usage count, and synchronization primitives.
/// The `state` mutex guards the current buffer and the in-flight list for coherent updates.
/// `compute_count` and `semaphore` remain accessible without locking for selection/coordination.
pub struct CommandBufferEntry {
    state: Mutex<EntryState>,
    compute_count: AtomicUsize,
    semaphore: Arc<CommandSemaphore>,
    /// Per-command-buffer profile state. `None` outside the profile feature
    /// build OR when no profiler is installed on the parent `Commands`.
    /// Populated lazily on the first profiled-encoder request, replaced on
    /// every `commit_swap_locked` so each command buffer has its own state.
    #[cfg(feature = "profile")]
    cb_profile: Mutex<Option<Arc<Mutex<CommandBufferProfile>>>>,
}

pub struct Commands {
    /// Maintains a pool of command buffers, allowing
    /// the pool to balance load across multiple buffers and improve GPU utilization.
    /// Can be shared across threads safely.
    pool: Vec<Arc<CommandBufferEntry>>,
    /// Single command queue for the entire device.
    command_queue: CommandQueue,
    /// The maximum amount of [compute command encoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc) per [command buffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc)
    compute_per_buffer: usize,
    /// Optional GPU profiler. When set, every encoder is constructed via
    /// `compute_command_encoder_profiled` so that begin/end timestamps land
    /// in a side `MTLCounterSampleBuffer`. Off-feature: field doesn't exist.
    #[cfg(feature = "profile")]
    profiler: RwLock<Option<Arc<MetalProfiler>>>,
    /// Serializes profiler lifecycle changes against encoder creation. Active
    /// encoders are still waited for via the entry semaphores during install / uninstall.
    #[cfg(feature = "profile")]
    profiler_lifecycle: Mutex<()>,
}

unsafe impl Send for Commands {}
unsafe impl Sync for Commands {}

impl Commands {
    pub fn new(command_queue: CommandQueue) -> Result<Self, MetalKernelError> {
        let compute_per_buffer = match std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER") {
            Ok(val) => val
                .parse()
                .unwrap_or(DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER),
            _ => DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER,
        };

        let pool_size = match std::env::var("CANDLE_METAL_COMMAND_POOL_SIZE") {
            Ok(val) => val
                .parse()
                .unwrap_or(DEFAULT_CANDLE_METAL_COMMAND_POOL_SIZE),
            _ => DEFAULT_CANDLE_METAL_COMMAND_POOL_SIZE,
        };

        let pool = (0..pool_size)
            .map(|_| Self::create_pool_entry(&command_queue))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            pool,
            command_queue,
            compute_per_buffer,
            #[cfg(feature = "profile")]
            profiler: RwLock::new(None),
            #[cfg(feature = "profile")]
            profiler_lifecycle: Mutex::new(()),
        })
    }

    /// Install a GPU profiler. All subsequently-constructed encoders will
    /// carry stage-boundary sample-buffer attachments; the addCompletedHandler
    /// resolves them off-thread when each command buffer finishes. Returns
    /// the previously-installed profiler, if any.
    #[cfg(feature = "profile")]
    pub fn install_profiler(
        &self,
        profiler: Option<Arc<MetalProfiler>>,
    ) -> Result<Option<Arc<MetalProfiler>>, MetalKernelError> {
        let _lifecycle = self.profiler_lifecycle.lock()?;

        // Drain all command buffers before any profiler lifecycle transition.
        // On first install this prevents mixed command buffers containing both
        // pre-install unprofiled work and post-install profiled work. On
        // replacement/removal it lets pending per-CB sample buffers install
        // completion handlers while the old profiler is still visible.
        self.flush_and_wait()?;

        // Be defensive for direct `Commands` users and for old buggy states:
        // no per-CB profile state should survive a profiler replacement.
        self.clear_entry_profiles()?;

        let mut g = self.profiler.write().map_err(|_| {
            MetalKernelError::FailedToCreateResource("profiler RwLock poisoned".into())
        })?;
        Ok(std::mem::replace(&mut *g, profiler))
    }

    #[cfg(feature = "profile")]
    fn clear_entry_profiles(&self) -> Result<(), MetalKernelError> {
        for entry in &self.pool {
            *entry.cb_profile.lock()? = None;
        }
        Ok(())
    }

    #[cfg(feature = "profile")]
    pub fn profiler(&self) -> Option<Arc<MetalProfiler>> {
        self.profiler.read().ok().and_then(|g| g.clone())
    }

    fn create_pool_entry(
        command_queue: &CommandQueue,
    ) -> Result<Arc<CommandBufferEntry>, MetalKernelError> {
        let semaphore = Arc::new(CommandSemaphore::new());
        let cb = create_command_buffer(command_queue, Arc::clone(&semaphore))?;

        Ok(Arc::new(CommandBufferEntry {
            state: Mutex::new(EntryState {
                current: cb,
                in_flight: Vec::new(),
            }),
            compute_count: AtomicUsize::new(0),
            semaphore,
            #[cfg(feature = "profile")]
            cb_profile: Mutex::new(None),
        }))
    }

    pub fn command_encoder(&self) -> Result<(bool, ComputeCommandEncoder), MetalKernelError> {
        #[cfg(feature = "profile")]
        let _lifecycle = self.profiler_lifecycle.lock()?;
        let entry = self.select_entry()?;
        #[cfg(feature = "profile")]
        {
            // Snapshot the installed profiler under read-lock; release before
            // touching the entry's mutexes to avoid lock ordering issues.
            let profiler_arc = self.profiler.read().ok().and_then(|g| g.clone());
            if let Some(profiler) = profiler_arc {
                return self.finalize_entry_profiled(entry, profiler);
            }
        }
        self.finalize_entry(entry, |cb| cb.compute_command_encoder())
    }

    /// Profiled variant of `finalize_entry`. Lazily allocates a per-CB profile
    /// on first use, claims a slot pair, and constructs an encoder via the
    /// descriptor variant with the sample buffer attached.
    #[cfg(feature = "profile")]
    fn finalize_entry_profiled(
        &self,
        entry: Arc<CommandBufferEntry>,
        profiler: Arc<MetalProfiler>,
    ) -> Result<(bool, ComputeCommandEncoder), MetalKernelError> {
        let mut state = entry.state.lock()?;
        let count = entry.compute_count.fetch_add(1, Ordering::Relaxed);
        let flush = count >= self.compute_per_buffer;
        if flush {
            self.commit_swap_locked(&entry, &mut state, 1)?;
        }

        // Lazy-init the per-command-buffer profile state for the *current* CB.
        let cb_profile_arc: Arc<Mutex<CommandBufferProfile>> = {
            let mut cbp_guard = entry.cb_profile.lock()?;
            if cbp_guard.is_none() {
                let new_profile = match profiler.new_command_buffer_profile() {
                    Ok(new_profile) => new_profile,
                    Err(e) => {
                        // We claimed this entry in `select_entry()` and
                        // incremented its compute count before trying to
                        // allocate the profile state. Roll that claim back so
                        // a transient sample-buffer allocation failure does
                        // not wedge this pool slot in `Encoding` forever.
                        if flush {
                            entry.compute_count.store(0, Ordering::Release);
                        } else {
                            entry.compute_count.fetch_sub(1, Ordering::Relaxed);
                        }
                        entry.semaphore.set_status(CommandStatus::Available);
                        return Err(MetalKernelError::FailedToCreateResource(format!(
                            "MTLCounterSampleBuffer alloc: {e}"
                        )));
                    }
                };
                *cbp_guard = Some(Arc::new(Mutex::new(new_profile)));
            }
            cbp_guard.as_ref().unwrap().clone()
        };

        let encoder = state.current.compute_command_encoder_profiled(
            cb_profile_arc,
            Arc::clone(&profiler),
            String::new(),
        );
        Ok((flush, encoder))
    }

    pub fn blit_command_encoder(&self) -> Result<(bool, BlitCommandEncoder), MetalKernelError> {
        #[cfg(feature = "profile")]
        let _lifecycle = self.profiler_lifecycle.lock()?;
        let entry = self.select_entry()?;
        self.finalize_entry(entry, |cb| cb.blit_command_encoder())
    }

    pub fn wait_until_completed(&self) -> Result<(), MetalKernelError> {
        self.flush_and_wait()
    }

    // Selects an entry from the pool using a two-phase strategy:
    /// 1. Try non-blocking: find any available buffer without waiting
    /// 2. Fallback: select the least-loaded buffer and wait for availability
    fn select_entry(&self) -> Result<Arc<CommandBufferEntry>, MetalKernelError> {
        // Phase 1: Try to find an available buffer without blocking
        for entry in &self.pool {
            if let Ok(mut status) = entry.semaphore.status.try_lock() {
                if matches!(*status, CommandStatus::Available) {
                    *status = CommandStatus::Encoding;
                    return Ok(Arc::clone(entry));
                }
            }
        }

        // Phase 2: Select the buffer with the most work and wait for it
        let entry = self
            .pool
            .iter()
            .max_by_key(|e| e.compute_count.load(Ordering::Acquire))
            .ok_or(MetalKernelError::FailedToCreateResource(
                "Command buffer pool is empty".to_string(),
            ))?;

        let entry = Arc::clone(entry);
        {
            let mut guard = entry
                .semaphore
                .wait_until(|s| matches!(s, CommandStatus::Available));
            *guard = CommandStatus::Encoding;
        }

        Ok(entry)
    }

    /// Creates an encoder from the selected entry, recycling the buffer if needed.
    /// When recycling, the old committed buffer is moved to `in_flight` so we can later wait on it.
    fn finalize_entry<F, E>(
        &self,
        entry: Arc<CommandBufferEntry>,
        create_encoder: F,
    ) -> Result<(bool, E), MetalKernelError>
    where
        F: FnOnce(&mut CommandBuffer) -> E,
    {
        let mut state = entry.state.lock()?;

        let count = entry.compute_count.fetch_add(1, Ordering::Relaxed);
        let flush = count >= self.compute_per_buffer;

        if flush {
            self.commit_swap_locked(&entry, &mut state, 1)?;
        }

        let encoder = create_encoder(&mut state.current);

        Ok((flush, encoder))
    }

    /// Flushes all buffers and waits for their completion.
    /// Commits any pending work on the current buffers, moves them to in-flight,
    /// then waits on all in-flight buffers including those from prior recycles.
    pub fn flush_and_wait(&self) -> Result<(), MetalKernelError> {
        for entry in &self.pool {
            // Under state lock, commit current if it has pending work and swap to a fresh one.
            let to_wait: Vec<CommandBuffer> = {
                // Ensure no active encoder is still encoding on this entry.
                let _guard = entry
                    .semaphore
                    .wait_until(|s| matches!(s, CommandStatus::Available));

                let mut state = entry.state.lock()?;

                if entry.compute_count.load(Ordering::Acquire) > 0 {
                    self.commit_swap_locked(entry, &mut state, 0)?;
                }

                // Drain `in_flight` into a local vec to wait without holding the lock.
                // Replaces `state.in_flight` with an empty vec and returns its previous contents.
                std::mem::take(&mut state.in_flight)
            };

            #[cfg(feature = "profile")]
            let profiler_arc = self.profiler.read().ok().and_then(|g| g.clone());

            for cb in to_wait {
                #[cfg(feature = "profile")]
                let wait_start_ns = profiler_arc.as_ref().map(|_| cpu_now_ns());
                #[cfg(feature = "profile")]
                let status_before = cb.status();

                Self::ensure_completed(&cb)?;

                #[cfg(feature = "profile")]
                if let (Some(profiler), Some(start_ns)) = (&profiler_arc, wait_start_ns) {
                    profiler.record_cpu_event(
                        Lane::CpuCommitWait,
                        "command_buffer.wait_until_completed",
                        start_ns,
                        cpu_now_ns(),
                        vec![
                            (
                                "status_before".into(),
                                ArgValue::String(Self::status_name(status_before).into()),
                            ),
                            (
                                "status_after".into(),
                                ArgValue::String(Self::status_name(cb.status()).into()),
                            ),
                        ],
                    );
                }
            }
        }

        Ok(())
    }

    /// Flushes all buffers without waiting for completion.
    /// Commits any pending work and moves current buffers to in-flight.
    pub fn flush(&self) -> Result<(), MetalKernelError> {
        for entry in &self.pool {
            let _guard = entry
                .semaphore
                .wait_until(|s| matches!(s, CommandStatus::Available));

            let mut state = entry.state.lock()?;

            if entry.compute_count.load(Ordering::Acquire) > 0 {
                self.commit_swap_locked(entry, &mut state, 0)?;
            }
        }

        Ok(())
    }

    /// Commit the current command buffer, swap in a fresh one, push the old into `in_flight`,
    /// and reset `compute_count` to `reset_to`.
    fn commit_swap_locked(
        &self,
        entry: &CommandBufferEntry,
        state: &mut EntryState,
        reset_to: usize,
    ) -> Result<(), MetalKernelError> {
        // If profiling is on AND this entry has a populated profile state,
        // install the completion handler before commit so the GPU's stage-
        // boundary timestamps are resolved off-thread once it finishes.
        // Then clear the entry's slot so the next CB starts fresh.
        #[cfg(feature = "profile")]
        let profiler_arc = self.profiler.read().ok().and_then(|g| g.clone());

        #[cfg(feature = "profile")]
        {
            let mut cbp_guard = entry.cb_profile.lock()?;
            let cb_profile_arc = cbp_guard.take();
            if let (Some(profiler), Some(cb_profile_arc)) = (&profiler_arc, cb_profile_arc) {
                state
                    .current
                    .install_profile_completion(cb_profile_arc, Arc::clone(profiler));
            }
        }

        #[cfg(feature = "profile")]
        let commit_start_ns = profiler_arc.as_ref().map(|_| cpu_now_ns());
        state.current.commit();
        #[cfg(feature = "profile")]
        if let (Some(profiler), Some(start_ns)) = (&profiler_arc, commit_start_ns) {
            profiler.record_cpu_event(
                Lane::CpuCommitWait,
                "command_buffer.commit",
                start_ns,
                cpu_now_ns(),
                vec![("reset_to".into(), ArgValue::U64(reset_to as u64))],
            );
        }

        let new_cb = create_command_buffer(&self.command_queue, Arc::clone(&entry.semaphore))?;
        let old_cb = std::mem::replace(&mut state.current, new_cb);
        state.in_flight.push(old_cb);
        entry.compute_count.store(reset_to, Ordering::Release);

        Ok(())
    }

    /// Human-readable name for an `MTLCommandBufferStatus` value. Used in
    /// trace event args so SQL queries see `"Completed"` rather than
    /// `MTLCommandBufferStatus(4)`.
    #[cfg(feature = "profile")]
    fn status_name(status: MTLCommandBufferStatus) -> &'static str {
        match status {
            MTLCommandBufferStatus::NotEnqueued => "NotEnqueued",
            MTLCommandBufferStatus::Enqueued => "Enqueued",
            MTLCommandBufferStatus::Committed => "Committed",
            MTLCommandBufferStatus::Scheduled => "Scheduled",
            MTLCommandBufferStatus::Completed => "Completed",
            MTLCommandBufferStatus::Error => "Error",
            _ => "Unknown",
        }
    }

    fn ensure_completed(cb: &CommandBuffer) -> Result<(), MetalKernelError> {
        match cb.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                cb.commit();
                cb.wait_until_completed();
            }
            MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled => {
                cb.wait_until_completed();
            }
            MTLCommandBufferStatus::Completed => {}
            MTLCommandBufferStatus::Error => {
                let msg = cb
                    .error()
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "unknown error".to_string());
                return Err(MetalKernelError::CommandBufferError(msg));
            }
            _ => unreachable!(),
        }

        Ok(())
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        // TODO: Avoid redundant allocation before drop
        let _ = self.flush();
    }
}
