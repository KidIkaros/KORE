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

// ── O(1) doubly-linked list arena used for LRU ordering ──────────────

/// Sentinel value indicating no link.
const NONE: usize = usize::MAX;

/// A node in the LRU doubly-linked list.
struct LruNode {
    key: String,
    prev: usize,
    next: usize,
}

/// Arena-allocated doubly-linked list for LRU ordering.
///
/// Supports O(1) insert-at-tail, remove-by-handle, and pop-from-head.
struct LruList {
    nodes: Vec<LruNode>,
    head: usize,
    tail: usize,
    free: Vec<usize>,
}

impl LruList {
    fn new() -> Self {
        Self { nodes: Vec::new(), head: NONE, tail: NONE, free: Vec::new() }
    }

    /// Insert a key at the tail (MRU position). Returns the node handle.
    fn push_back(&mut self, key: String) -> usize {
        let idx = if let Some(free_idx) = self.free.pop() {
            self.nodes[free_idx] = LruNode { key, prev: self.tail, next: NONE };
            free_idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(LruNode { key, prev: self.tail, next: NONE });
            idx
        };

        if self.tail != NONE {
            self.nodes[self.tail].next = idx;
        } else {
            self.head = idx;
        }
        self.tail = idx;
        idx
    }

    /// Remove a node by handle. O(1).
    fn remove(&mut self, idx: usize) {
        let prev = self.nodes[idx].prev;
        let next = self.nodes[idx].next;

        if prev != NONE {
            self.nodes[prev].next = next;
        } else {
            self.head = next;
        }
        if next != NONE {
            self.nodes[next].prev = prev;
        } else {
            self.tail = prev;
        }

        self.nodes[idx].prev = NONE;
        self.nodes[idx].next = NONE;
        self.free.push(idx);
    }

    /// Move an existing node to the tail (MRU). O(1).
    fn move_to_back(&mut self, idx: usize) {
        if idx == self.tail {
            return; // already MRU
        }
        self.remove(idx);
        // Re-insert at tail (reuse the slot directly)
        self.nodes[idx].prev = self.tail;
        self.nodes[idx].next = NONE;
        // Reclaim from free list since we're reusing it
        if let Some(pos) = self.free.iter().position(|&f| f == idx) {
            self.free.swap_remove(pos);
        }
        if self.tail != NONE {
            self.nodes[self.tail].next = idx;
        } else {
            self.head = idx;
        }
        self.tail = idx;
    }

    /// Pop the head (LRU) node. Returns the key. O(1).
    fn pop_front(&mut self) -> Option<String> {
        if self.head == NONE {
            return None;
        }
        let idx = self.head;
        let key = self.nodes[idx].key.clone();
        self.remove(idx);
        Some(key)
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.nodes.len() - self.free.len()
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.head = NONE;
        self.tail = NONE;
        self.free.clear();
    }
}

// ── LayerCache ───────────────────────────────────────────────────────

/// LRU cache for layer state dicts in RAM.
///
/// Uses an arena-backed doubly-linked list for O(1) get, put, and evict.
pub struct LayerCache {
    inner: Mutex<CacheInner>,
}

impl std::fmt::Debug for LayerCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.lock();
        f.debug_struct("LayerCache")
            .field("entries", &inner.map.len())
            .field("max_layers", &inner.max_layers)
            .field("max_bytes", &inner.max_bytes)
            .field("current_bytes", &inner.current_bytes)
            .finish()
    }
}

struct CacheInner {
    /// Map from layer name → (weights, size, LRU node handle).
    map: HashMap<String, CacheValue>,
    /// LRU ordering list.
    lru: LruList,
    max_layers: usize,
    max_bytes: usize,
    current_bytes: usize,
    hits: u64,
    misses: u64,
}

struct CacheValue {
    weights: LayerWeights,
    size_bytes: usize,
    lru_handle: usize,
}

impl LayerCache {
    /// Create a new layer cache with the given maximum number of layers.
    pub fn new(max_layers: usize) -> Self {
        Self {
            inner: Mutex::new(CacheInner {
                map: HashMap::new(),
                lru: LruList::new(),
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
                map: HashMap::new(),
                lru: LruList::new(),
                max_layers: 256,
                max_bytes: budget,
                current_bytes: 0,
                hits: 0,
                misses: 0,
            }),
        }
    }

    /// Check if a layer is in the cache.
    pub fn has(&self, layer_name: &str) -> bool {
        self.inner.lock().map.contains_key(layer_name)
    }

    /// Get a layer from the cache, moving it to the most-recently-used position.
    ///
    /// O(1) — linked-list node is moved to tail without shifting other entries.
    pub fn get(&self, layer_name: &str) -> Option<LayerWeights> {
        let mut inner = self.inner.lock();

        // Extract handle + weights first to release the immutable borrow on `map`.
        let found = inner.map.get(layer_name).map(|val| (val.lru_handle, val.weights.clone()));

        if let Some((handle, weights)) = found {
            inner.hits += 1;
            inner.lru.move_to_back(handle);
            Some(weights)
        } else {
            inner.misses += 1;
            None
        }
    }

    /// Insert a layer into the cache, evicting LRU entries if necessary.
    ///
    /// O(1) amortized — eviction pops the linked-list head and removes from the map.
    pub fn put(&self, layer_name: String, weights: LayerWeights) {
        let size = estimate_weights_size(&weights);
        let mut inner = self.inner.lock();

        if inner.map.contains_key(&layer_name) {
            return;
        }

        // Evict LRU entries until we have room
        while needs_eviction(&inner, size) {
            if let Some(evict_key) = inner.lru.pop_front() {
                if let Some(evicted) = inner.map.remove(&evict_key) {
                    inner.current_bytes = inner.current_bytes.saturating_sub(evicted.size_bytes);
                }
            } else {
                break;
            }
        }

        let handle = inner.lru.push_back(layer_name.clone());
        inner.map.insert(layer_name, CacheValue { weights, size_bytes: size, lru_handle: handle });
        inner.current_bytes += size;
    }

    /// Clear the entire cache.
    pub fn clear(&self) {
        let mut inner = self.inner.lock();
        inner.map.clear();
        inner.lru.clear();
        inner.current_bytes = 0;
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let inner = self.inner.lock();
        let total = inner.hits + inner.misses;
        CacheStats {
            cached_layers: inner.map.len(),
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
    if inner.map.len() >= inner.max_layers {
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

    #[test]
    fn test_lru_list_ops() {
        let mut lru = LruList::new();
        assert_eq!(lru.len(), 0);

        let a = lru.push_back("a".into());
        let b = lru.push_back("b".into());
        let _c = lru.push_back("c".into());
        assert_eq!(lru.len(), 3);

        // Pop front should give LRU ("a")
        assert_eq!(lru.pop_front().unwrap(), "a");
        assert_eq!(lru.len(), 2);

        // Move b to back, then pop front should give "c"
        lru.move_to_back(b);
        assert_eq!(lru.pop_front().unwrap(), "c");
        assert_eq!(lru.pop_front().unwrap(), "b");
        assert_eq!(lru.len(), 0);

        // Reuse freed slots
        let _d = lru.push_back("d".into());
        assert_eq!(lru.len(), 1);
        assert!(a < lru.nodes.len()); // slot was reused
    }
}
