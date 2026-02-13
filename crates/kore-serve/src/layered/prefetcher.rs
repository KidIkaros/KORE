//! Async double-buffered layer prefetcher.
//!
//! Overlaps disk I/O with GPU computation by loading upcoming layers
//! in background threads while the current layer is being processed.
//!
//! ```text
//! Time →
//! Layer N:   [====COMPUTE====]
//! Layer N+1:      [LOAD][DECOMPRESS][GPU_XFER]
//! Layer N+2:                  [LOAD][DECOMPRESS][GPU_XFER]
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::Mutex;
use tokio::sync::oneshot;

use super::cache::{LayerCache, LayerWeights};

/// Async double-buffered layer prefetcher.
///
/// Uses `tokio::spawn_blocking` to load layers from disk in background
/// threads while the main thread computes on the current layer.
pub struct LayerPrefetcher {
    /// Directory containing per-layer shard files.
    shard_dir: PathBuf,
    /// All layer names in forward-pass order.
    layer_names: Vec<String>,
    /// Number of layers to prefetch ahead.
    lookahead: usize,
    /// Shared layer cache (checked before loading from disk).
    cache: Arc<LayerCache>,
    /// In-flight prefetch results.
    pending: Arc<Mutex<HashMap<usize, PendingLayer>>>,
    /// Profiling data.
    profile: Arc<Mutex<ProfileData>>,
    /// Whether profiling is enabled.
    profiling: bool,
}

enum PendingLayer {
    /// Load is in progress; receiver will deliver the result.
    Loading(oneshot::Receiver<Result<LayerWeights, String>>),
}

#[derive(Default)]
struct ProfileData {
    load_times: Vec<(usize, f64)>,
    wait_times: Vec<(usize, f64)>,
}

impl LayerPrefetcher {
    /// Create a new prefetcher.
    pub fn new(
        shard_dir: PathBuf,
        layer_names: Vec<String>,
        lookahead: usize,
        cache: Arc<LayerCache>,
        profiling: bool,
    ) -> Self {
        Self {
            shard_dir,
            layer_names,
            lookahead,
            cache,
            pending: Arc::new(Mutex::new(HashMap::new())),
            profile: Arc::new(Mutex::new(ProfileData::default())),
            profiling,
        }
    }

    /// Start prefetching the first `lookahead` layers.
    pub fn start(&self) {
        for i in 0..self.lookahead.min(self.layer_names.len()) {
            self.submit_load(i);
        }
    }

    /// Get layer weights for index `idx`, blocking until ready.
    ///
    /// Automatically triggers prefetch of upcoming layers.
    pub async fn get_layer(&self, idx: usize) -> Result<LayerWeights, String> {
        let layer_name = self.layer_names.get(idx)
            .ok_or_else(|| format!("layer index {} out of range ({})", idx, self.layer_names.len()))?;

        // Check cache first
        if let Some(weights) = self.cache.get(layer_name) {
            // Still trigger prefetch of upcoming layers
            self.prefetch_ahead(idx);
            return Ok(weights);
        }

        let t_start = std::time::Instant::now();

        // Check if we have a pending load
        let pending = {
            let mut pending_map = self.pending.lock();
            pending_map.remove(&idx)
        };

        let weights = match pending {
            Some(PendingLayer::Loading(rx)) => {
                rx.await.map_err(|_| "prefetch channel closed".to_string())??
            }
            None => {
                // Not prefetched — load synchronously
                load_layer_from_disk(&self.shard_dir, layer_name)?
            }
        };

        if self.profiling {
            let elapsed = t_start.elapsed().as_secs_f64();
            self.profile.lock().wait_times.push((idx, elapsed));
        }

        // Cache the loaded weights
        self.cache.put(layer_name.clone(), weights.clone());

        // Trigger prefetch of upcoming layers
        self.prefetch_ahead(idx);

        Ok(weights)
    }

    /// Release resources for a layer (remove from pending buffer).
    pub fn release(&self, idx: usize) {
        self.pending.lock().remove(&idx);
    }

    /// Get profiling summary.
    pub fn profile_summary(&self) -> Option<PrefetcherStats> {
        if !self.profiling {
            return None;
        }

        let data = self.profile.lock();
        let load_times: Vec<f64> = data.load_times.iter().map(|(_, t)| *t).collect();
        let wait_times: Vec<f64> = data.wait_times.iter().map(|(_, t)| *t).collect();

        let avg = |v: &[f64]| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
        let sum = |v: &[f64]| v.iter().sum::<f64>();

        let total_load = sum(&load_times);
        let total_wait = sum(&wait_times);
        let overlap = if total_load > 0.0 { 1.0 - (total_wait / total_load) } else { 0.0 };

        Some(PrefetcherStats {
            avg_load_time: avg(&load_times),
            avg_wait_time: avg(&wait_times),
            total_load_time: total_load,
            total_wait_time: total_wait,
            overlap_efficiency: overlap.max(0.0),
        })
    }

    fn prefetch_ahead(&self, current_idx: usize) {
        for ahead in 1..=self.lookahead {
            self.submit_load(current_idx + ahead);
        }
    }

    fn submit_load(&self, idx: usize) {
        if idx >= self.layer_names.len() {
            return;
        }

        let mut pending_map = self.pending.lock();
        if pending_map.contains_key(&idx) {
            return; // Already submitted
        }

        let layer_name = self.layer_names[idx].clone();

        // Skip if already cached
        if self.cache.has(&layer_name) {
            return;
        }

        let (tx, rx) = oneshot::channel();
        pending_map.insert(idx, PendingLayer::Loading(rx));

        // Verify a Tokio runtime is available before spawning.
        // If not, send an error immediately so callers never hang.
        let handle = match tokio::runtime::Handle::try_current() {
            Ok(h) => h,
            Err(_) => {
                let _ = tx.send(Err("no async runtime available for prefetch".into()));
                return;
            }
        };

        let shard_dir = self.shard_dir.clone();
        let profiling = self.profiling;
        let profile = self.profile.clone();

        // Capture the JoinHandle so task panics are detected.
        // If the outer spawn panics or is cancelled, `tx` is dropped and
        // the receiver gets a clear error instead of hanging indefinitely.
        let _join = handle.spawn(async move {
            let result = match tokio::task::spawn_blocking(move || {
                let t_start = std::time::Instant::now();
                let weights = load_layer_from_disk(&shard_dir, &layer_name);
                if profiling {
                    let elapsed = t_start.elapsed().as_secs_f64();
                    profile.lock().load_times.push((idx, elapsed));
                }
                weights
            })
            .await
            {
                Ok(inner) => inner,
                Err(e) => Err(format!("prefetch task for layer {idx} panicked or was cancelled: {e}")),
            };

            // If the receiver has been dropped (e.g. caller timed out), this is a no-op.
            let _ = tx.send(result);
        });
    }
}

/// Profiling statistics for the prefetcher.
#[derive(Debug, Clone)]
pub struct PrefetcherStats {
    pub avg_load_time: f64,
    pub avg_wait_time: f64,
    pub total_load_time: f64,
    pub total_wait_time: f64,
    pub overlap_efficiency: f64,
}

impl std::fmt::Display for PrefetcherStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Prefetcher: avg load {:.3}s, avg wait {:.3}s, overlap {:.1}%",
            self.avg_load_time,
            self.avg_wait_time,
            self.overlap_efficiency * 100.0,
        )
    }
}

/// Load a single layer's weights from a safetensors shard file on disk.
///
/// Expects the file at `<shard_dir>/<layer_name>.safetensors`.
pub fn load_layer_from_disk(shard_dir: &Path, layer_name: &str) -> Result<LayerWeights, String> {
    let safe_name = sanitize_layer_name(layer_name);
    let path = shard_dir.join(format!("{safe_name}.safetensors"));

    if !path.exists() {
        return Err(format!("shard file not found: {}", path.display()));
    }

    let file_bytes = std::fs::read(&path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;

    let tensors = safetensors::SafeTensors::deserialize(&file_bytes)
        .map_err(|e| format!("failed to parse safetensors {}: {e}", path.display()))?;

    let mut weights = Vec::new();
    for (name, view) in tensors.tensors() {
        weights.push((name.to_string(), view.data().to_vec()));
    }

    Ok(Arc::new(weights))
}

/// Sanitize a layer name into a filesystem-safe filename component.
///
/// Replaces dots, slashes, colons, and other unsafe characters with underscores.
fn sanitize_layer_name(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            '.' | '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | ' ' => '_',
            _ => c,
        })
        .collect()
}
