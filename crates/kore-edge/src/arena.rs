//! Arena allocator for zero-alloc steady-state inference.
//!
//! Pre-allocates a single contiguous buffer, bump-allocates during inference,
//! and resets between runs. No per-tensor heap allocation in the hot path.

/// A simple bump-pointer arena allocator.
///
/// Allocations are 16-byte aligned for SIMD compatibility.
/// Call `reset()` between inference runs to reuse memory.
pub struct Arena {
    buf: Vec<u8>,
    offset: usize,
}

const ARENA_ALIGN: usize = 16;

impl Arena {
    /// Create an arena with the given capacity in bytes.
    pub fn new(capacity: usize) -> Self {
        Self {
            buf: vec![0u8; capacity],
            offset: 0,
        }
    }

    /// Allocate `nbytes` from the arena, returning a mutable slice.
    /// Returns `None` if the arena is exhausted.
    pub fn alloc(&mut self, nbytes: usize) -> Option<&mut [u8]> {
        let aligned_offset = align_up(self.offset, ARENA_ALIGN);
        let end = aligned_offset + nbytes;
        if end > self.buf.len() {
            return None;
        }
        self.offset = end;
        Some(&mut self.buf[aligned_offset..end])
    }

    /// Allocate space for `n` f32 values, returning a mutable f32 slice.
    pub fn alloc_f32(&mut self, n: usize) -> Option<&mut [f32]> {
        let nbytes = n * 4;
        let aligned_offset = align_up(self.offset, ARENA_ALIGN);
        let end = aligned_offset + nbytes;
        if end > self.buf.len() {
            return None;
        }
        self.offset = end;
        let slice = &mut self.buf[aligned_offset..end];
        let ptr = slice.as_mut_ptr() as *mut f32;
        Some(unsafe { std::slice::from_raw_parts_mut(ptr, n) })
    }

    /// Allocate and zero-fill space for `n` f32 values.
    pub fn alloc_f32_zeroed(&mut self, n: usize) -> Option<&mut [f32]> {
        let s = self.alloc_f32(n)?;
        for v in s.iter_mut() {
            *v = 0.0;
        }
        Some(s)
    }

    /// Reset the arena for reuse. Does not deallocate.
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Current bytes used.
    pub fn used(&self) -> usize {
        self.offset
    }

    /// Total capacity in bytes.
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Remaining bytes available.
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.offset
    }
}

fn align_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_alloc() {
        let mut arena = Arena::new(1024);
        let s = arena.alloc(100).unwrap();
        assert_eq!(s.len(), 100);
        assert!(arena.used() >= 100);
    }

    #[test]
    fn test_alloc_f32() {
        let mut arena = Arena::new(1024);
        let s = arena.alloc_f32(10).unwrap();
        assert_eq!(s.len(), 10);
        s[0] = 42.0;
        assert_eq!(s[0], 42.0);
    }

    #[test]
    fn test_alloc_f32_zeroed() {
        let mut arena = Arena::new(1024);
        let s = arena.alloc_f32_zeroed(10).unwrap();
        assert!(s.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_reset() {
        let mut arena = Arena::new(256);
        arena.alloc(200).unwrap();
        assert!(arena.used() >= 200);
        arena.reset();
        assert_eq!(arena.used(), 0);
        // Can allocate again
        arena.alloc(200).unwrap();
    }

    #[test]
    fn test_exhaustion() {
        let mut arena = Arena::new(64);
        assert!(arena.alloc(32).is_some());
        assert!(arena.alloc(64).is_none()); // not enough left
    }

    #[test]
    fn test_alignment() {
        let mut arena = Arena::new(1024);
        arena.alloc(1).unwrap(); // 1 byte
        let s = arena.alloc_f32(1).unwrap(); // should be aligned
        let ptr = s.as_ptr() as usize;
        assert_eq!(ptr % ARENA_ALIGN, 0, "f32 alloc not aligned");
    }

    #[test]
    fn test_multiple_allocs() {
        let mut arena = Arena::new(4096);
        for _ in 0..10 {
            let s = arena.alloc_f32(64).unwrap();
            assert_eq!(s.len(), 64);
        }
        assert!(arena.used() <= 4096);
    }
}
