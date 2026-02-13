//! LRU RAM cache for layer weights during layered inference.
//!
//! During autoregressive generation, every token requires a full forward pass
//! through all layers. Without caching, each layer must be reloaded from disk
//! for every token. This module caches layer weights in RAM to avoid redundant
//! disk I/O.

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::Mutex;

/// A single cached layer: mapping from parameter name to raw weight bytes.
///
/// Wrapped in `Arc` to allow cheap cloning when layers are shared between
/// the cache, prefetcher, and engine without duplicating multi-GB buffers.
pub type LayerWeights = Arc<Vec<(String, Vec<u8>)>>;

/// LRU cache for layer state dicts in RAM.
///
/// Keeps up to `max_layers` layers in memory. When full, evicts the
/// least-recently-used layer to make room for new ones.
pub struct LayerCache {
    inner: Mutex<CacheInner>,
}

impl std::fmt::Debug for LayerCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.lock();
        f.debug_struct("LayerCache")
            .field("entries", &inner.entries.len())
            .field("max_layers", &inner.max_layers)
            .field("max_bytes", &inner.max_bytes)
            .field("current_bytes", &inner.current_bytes)
            .finish()
    }
}

struct CacheInner {
    /// Layers stored in access order (most recent at the end).
    entries: Vec<CacheEntry>,
    /// Map from layer name to index in `entries`.
    index: HashMap<String, usize>,
    /// Maximum number of layers to cache.
    max_layers: usize,
    /// Maximum bytes to use for cache (0 = use max_layers only).
    max_bytes: usize,
    /// Current total size in bytes.
    current_bytes: usize,
    /// Cache hit count.
    hits: u64,
    /// Cache miss count.
    misses: u64,
}

struct CacheEntry {
    name: String,
    weights: LayerWeights,
    size_bytes: usize,
}

impl LayerCache {
    /// Create a new layer cache with the given maximum number of layers.
    pub fn new(max_layers: usize) -> Self {
        Self {
            inner: Mutex::new(CacheInner {
                entries: Vec::new(),
                index: HashMap::new(),
                max_layers,
                max_bytes: 0,
                current_bytes: 0,
                hits: 0,
                misses: 0,
            }),
        }
    }

    /// Create a cache that auto-sizes based on available system RAM.
    ///
    /// `memory_fraction` is the fraction of available RAM to use (0.0–1.0).
    pub fn adaptive(memory_fraction: f32) -> Self {
        let available = available_ram_bytes();
        let budget = (available as f64 * memory_fraction as f64) as usize;

        tracing::info!(
            "LayerCache: {:.1} GB available, budgeting {:.1} GB ({:.0}%)",
            available as f64 / 1e9,
            budget as f64 / 1e9,
            memory_fraction * 100.0,
        );

        Self {
            inner: Mutex::new(CacheInner {
                entries: Vec::new(),
                index: HashMap::new(),
                max_layers: 256, // generous upper bound
                max_bytes: budget,
                current_bytes: 0,
                hits: 0,
                misses: 0,
            }),
        }
    }

    /// Check if a layer is in the cache.
    pub fn has(&self, layer_name: &str) -> bool {
        let inner = self.inner.lock();
        inner.index.contains_key(layer_name)
    }

    /// Get a layer from the cache, moving it to the most-recently-used position.
    ///
    /// Returns `None` on cache miss.
    pub fn get(&self, layer_name: &str) -> Option<LayerWeights> {
        let mut inner = self.inner.lock();

        if let Some(&idx) = inner.index.get(layer_name) {
            inner.hits += 1;
            // Move to end (most recently used)
            let entry = inner.entries.remove(idx);
            let weights = entry.weights.clone();
            let new_idx = inner.entries.len();
            inner.entries.push(entry);
            // Incremental index update: only entries that shifted down
            let shifted: Vec<(String, usize)> = (idx..new_idx)
                .map(|i| (inner.entries[i].name.clone(), i))
                .collect();
            for (name, i) in shifted {
                inner.index.insert(name, i);
            }
            inner.index.insert(layer_name.to_string(), new_idx);
            Some(weights)
        } else {
            inner.misses += 1;
            None
        }
    }

    /// Insert a layer into the cache, evicting LRU entries if necessary.
    pub fn put(&self, layer_name: String, weights: LayerWeights) {
        let size = estimate_weights_size(&weights);
        let mut inner = self.inner.lock();

        // Already cached? Move to end.
        if inner.index.contains_key(&layer_name) {
            return;
        }

        // Evict until we have room
        while needs_eviction(&inner, size) {
            if inner.entries.is_empty() {
                break;
            }
            let evicted = inner.entries.remove(0);
            inner.current_bytes = inner.current_bytes.saturating_sub(evicted.size_bytes);
            inner.index.remove(&evicted.name);
            // Incremental index update: all entries shifted down by 1
            let shifted: Vec<(String, usize)> = inner.entries.iter()
                .enumerate()
                .map(|(i, e)| (e.name.clone(), i))
                .collect();
            for (name, i) in shifted {
                inner.index.insert(name, i);
            }
        }

        // Insert
        let entry = CacheEntry {
            name: layer_name.clone(),
            weights,
            size_bytes: size,
        };
        inner.entries.push(entry);
        let len = inner.entries.len();
        inner.index.insert(layer_name, len - 1);
        inner.current_bytes += size;
    }

    /// Clear the entire cache.
    pub fn clear(&self) {
        let mut inner = self.inner.lock();
        inner.entries.clear();
        inner.index.clear();
        inner.current_bytes = 0;
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let inner = self.inner.lock();
        let total = inner.hits + inner.misses;
        CacheStats {
            cached_layers: inner.entries.len(),
            max_layers: inner.max_layers,
            current_bytes: inner.current_bytes,
            max_bytes: inner.max_bytes,
            hits: inner.hits,
            misses: inner.misses,
            hit_rate: if total > 0 { inner.hits as f64 / total as f64 } else { 0.0 },
        }
    }
}

/// Cache statistics for monitoring.
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub cached_layers: usize,
    pub max_layers: usize,
    pub current_bytes: usize,
    pub max_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LayerCache: {}/{} layers, {:.1} MB, hit rate {:.1}% ({} hits, {} misses)",
            self.cached_layers,
            self.max_layers,
            self.current_bytes as f64 / (1024.0 * 1024.0),
            self.hit_rate * 100.0,
            self.hits,
            self.misses,
        )
    }
}

fn needs_eviction(inner: &CacheInner, new_size: usize) -> bool {
    if inner.entries.len() >= inner.max_layers {
        return true;
    }
    if inner.max_bytes > 0 && (inner.current_bytes + new_size) > inner.max_bytes {
        return true;
    }
    false
}

fn estimate_weights_size(weights: &LayerWeights) -> usize {
    weights.as_ref().iter().map(|(name, data)| name.len() + data.len()).sum()
}

fn available_ram_bytes() -> usize {
    let sys = sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::new().with_memory(sysinfo::MemoryRefreshKind::everything()),
    );
    sys.available_memory() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_put_get() {
        let cache = LayerCache::new(2);

        let w1 = Arc::new(vec![("w.weight".into(), vec![1u8; 100])]);
        let w2 = Arc::new(vec![("w.weight".into(), vec![2u8; 100])]);
        let w3 = Arc::new(vec![("w.weight".into(), vec![3u8; 100])]);

        cache.put("layer.0".into(), w1.clone());
        cache.put("layer.1".into(), w2.clone());

        assert!(cache.has("layer.0"));
        assert!(cache.has("layer.1"));

        // Eviction: adding layer.2 should evict layer.0 (LRU)
        cache.put("layer.2".into(), w3.clone());
        assert!(!cache.has("layer.0"));
        assert!(cache.has("layer.1"));
        assert!(cache.has("layer.2"));
    }

    #[test]
    fn test_cache_lru_order() {
        let cache = LayerCache::new(2);

        let w1 = Arc::new(vec![("w".into(), vec![1u8; 10])]);
        let w2 = Arc::new(vec![("w".into(), vec![2u8; 10])]);
        let w3 = Arc::new(vec![("w".into(), vec![3u8; 10])]);

        cache.put("a".into(), w1);
        cache.put("b".into(), w2);

        // Access "a" to make it MRU
        let _ = cache.get("a");

        // Now "b" is LRU — should be evicted
        cache.put("c".into(), w3);
        assert!(cache.has("a"));
        assert!(!cache.has("b"));
        assert!(cache.has("c"));
    }

    #[test]
    fn test_cache_stats() {
        let cache = LayerCache::new(4);
        cache.put("x".into(), Arc::new(vec![("w".into(), vec![0u8; 50])]));

        let _ = cache.get("x"); // hit
        let _ = cache.get("y"); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 0.5).abs() < 1e-6);
    }
}
