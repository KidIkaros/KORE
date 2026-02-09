//! Trit-addressed memory — variable-length ternary data container.
//!
//! Ported from `btes/src/memory.c`. Provides `TernaryFrame` for storing
//! packed ternary data with random access and bulk word conversion.
//! Rust `Drop` replaces manual `btes_frame_free`.

use crate::encoder::{Trit, encode_trits, decode_trits};
use crate::vtalu::TernaryWord64;

/// Packing mode for ternary storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackingMode {
    /// Base-243: 5 trits per byte (1.58 bits/trit). Default.
    Ternary,
    /// 4 trits per byte (2 bits each). Faster unpack on CPU.
    Quaternary,
}

/// Variable-length packed ternary data container.
///
/// Automatically freed on drop (no manual deallocation needed).
#[derive(Debug, Clone)]
pub struct TernaryFrame {
    data: Vec<u8>,
    trit_count: usize,
    mode: PackingMode,
}

impl TernaryFrame {
    /// Allocate a new frame with all trits set to zero.
    pub fn new(trit_count: usize, mode: PackingMode) -> Self {
        let byte_count = match mode {
            PackingMode::Ternary => (trit_count + 4) / 5,
            PackingMode::Quaternary => (trit_count + 3) / 4,
        };

        let fill_byte = match mode {
            // All-zero trits in base-243: encode_trits([Zero;5]) = 121
            PackingMode::Ternary => encode_trits(&[Trit::Zero; 5]),
            // All-zero in quaternary: 0x00
            PackingMode::Quaternary => 0,
        };

        Self {
            data: vec![fill_byte; byte_count],
            trit_count,
            mode,
        }
    }

    /// Allocate using default ternary packing.
    pub fn new_ternary(trit_count: usize) -> Self {
        Self::new(trit_count, PackingMode::Ternary)
    }

    /// Number of logical trits stored.
    pub fn trit_count(&self) -> usize {
        self.trit_count
    }

    /// Physical storage size in bytes.
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }

    /// Packing mode.
    pub fn mode(&self) -> PackingMode {
        self.mode
    }

    /// Get a single trit by index.
    pub fn get_trit(&self, index: usize) -> Trit {
        if index >= self.trit_count {
            return Trit::Zero;
        }

        match self.mode {
            PackingMode::Ternary => {
                let tryte_idx = index / 5;
                let trit_off = index % 5;
                let trits = decode_trits(self.data[tryte_idx]);
                trits[trit_off]
            }
            PackingMode::Quaternary => {
                let byte_idx = index / 4;
                let bit_off = (index % 4) * 2;
                let two_bits = (self.data[byte_idx] >> bit_off) & 0x3;
                match two_bits {
                    0 => Trit::Zero,
                    1 => Trit::Neg,  // maps to -1
                    2 => Trit::Pos,  // maps to +1
                    _ => Trit::Zero, // 3 is unused
                }
            }
        }
    }

    /// Set a single trit by index.
    pub fn set_trit(&mut self, index: usize, value: Trit) {
        if index >= self.trit_count {
            return;
        }

        match self.mode {
            PackingMode::Ternary => {
                let tryte_idx = index / 5;
                let trit_off = index % 5;
                let mut trits = decode_trits(self.data[tryte_idx]);
                trits[trit_off] = value;
                self.data[tryte_idx] = encode_trits(&trits);
            }
            PackingMode::Quaternary => {
                let byte_idx = index / 4;
                let bit_off = (index % 4) * 2;
                let mask = !(0x3u8 << bit_off);
                let val_bits: u8 = match value {
                    Trit::Zero => 0,
                    Trit::Neg => 1,
                    Trit::Pos => 2,
                };
                self.data[byte_idx] = (self.data[byte_idx] & mask) | (val_bits << bit_off);
            }
        }
    }

    /// Fill a range of trits with a single value.
    pub fn fill(&mut self, start: usize, count: usize, value: Trit) {
        let end = (start + count).min(self.trit_count);
        for i in start..end {
            self.set_trit(i, value);
        }
    }

    /// Convert a range of trits into `TernaryWord64` values (64 trits each).
    pub fn to_words(&self, start_trit: usize, max_words: usize) -> Vec<TernaryWord64> {
        let mut words = Vec::with_capacity(max_words);
        let mut current = start_trit;

        while words.len() < max_words && current < self.trit_count {
            let mut trits = [0i8; 64];
            for i in 0..64 {
                if current + i >= self.trit_count {
                    break;
                }
                trits[i] = self.get_trit(current + i) as i8;
            }
            words.push(TernaryWord64::from_trits(&trits));
            current += 64;
        }

        words
    }

    /// Write `TernaryWord64` values back into the frame.
    pub fn from_words(&mut self, words: &[TernaryWord64], start_trit: usize) -> usize {
        let mut current = start_trit;
        let mut written = 0;

        for word in words {
            let trits = word.to_trits();
            for &t in &trits {
                if current >= self.trit_count {
                    return written;
                }
                let trit = match t {
                    1 => Trit::Pos,
                    -1 => Trit::Neg,
                    _ => Trit::Zero,
                };
                self.set_trit(current, trit);
                current += 1;
                written += 1;
            }
        }

        written
    }

    /// Copy trits from another frame.
    pub fn copy_from(&mut self, dst_start: usize, src: &TernaryFrame, src_start: usize, count: usize) -> usize {
        let mut copied = 0;
        for i in 0..count {
            if src_start + i >= src.trit_count || dst_start + i >= self.trit_count {
                break;
            }
            self.set_trit(dst_start + i, src.get_trit(src_start + i));
            copied += 1;
        }
        copied
    }

    /// Check equality with another frame.
    pub fn eq_frame(&self, other: &TernaryFrame) -> bool {
        if self.trit_count != other.trit_count {
            return false;
        }
        for i in 0..self.trit_count {
            if self.get_trit(i) != other.get_trit(i) {
                return false;
            }
        }
        true
    }

    /// Count non-zero trits.
    pub fn popcount(&self) -> usize {
        (0..self.trit_count)
            .filter(|&i| self.get_trit(i) != Trit::Zero)
            .count()
    }

    /// Resize the frame, preserving existing data where possible.
    pub fn resize(&mut self, new_trit_count: usize) {
        let new_byte_count = match self.mode {
            PackingMode::Ternary => (new_trit_count + 4) / 5,
            PackingMode::Quaternary => (new_trit_count + 3) / 4,
        };

        let fill_byte = match self.mode {
            PackingMode::Ternary => encode_trits(&[Trit::Zero; 5]),
            PackingMode::Quaternary => 0,
        };

        if new_byte_count > self.data.len() {
            self.data.resize(new_byte_count, fill_byte);
        } else {
            self.data.truncate(new_byte_count);
        }
        self.trit_count = new_trit_count;
    }

    /// Get raw packed data.
    pub fn raw_data(&self) -> &[u8] {
        &self.data
    }
}

impl PartialEq for TernaryFrame {
    fn eq(&self, other: &Self) -> bool {
        self.eq_frame(other)
    }
}

impl std::fmt::Display for TernaryFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let max_display = 16;
        write!(f, "[")?;
        for i in 0..self.trit_count.min(max_display) {
            if i > 0 { write!(f, ", ")?; }
            match self.get_trit(i) {
                Trit::Neg => write!(f, "-1")?,
                Trit::Zero => write!(f, " 0")?,
                Trit::Pos => write!(f, "+1")?,
            }
        }
        if self.trit_count > max_display {
            write!(f, ", ... ({} more)", self.trit_count - max_display)?;
        }
        write!(f, "] ({} trits, {} bytes)", self.trit_count, self.data.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_ternary() {
        let frame = TernaryFrame::new_ternary(100);
        assert_eq!(frame.trit_count(), 100);
        assert_eq!(frame.byte_size(), 20); // ceil(100/5)
        assert_eq!(frame.mode(), PackingMode::Ternary);
    }

    #[test]
    fn test_new_quaternary() {
        let frame = TernaryFrame::new(100, PackingMode::Quaternary);
        assert_eq!(frame.trit_count(), 100);
        assert_eq!(frame.byte_size(), 25); // ceil(100/4)
    }

    #[test]
    fn test_get_set_ternary() {
        let mut frame = TernaryFrame::new_ternary(10);
        assert_eq!(frame.get_trit(0), Trit::Zero);

        frame.set_trit(0, Trit::Pos);
        frame.set_trit(1, Trit::Neg);
        frame.set_trit(9, Trit::Pos);

        assert_eq!(frame.get_trit(0), Trit::Pos);
        assert_eq!(frame.get_trit(1), Trit::Neg);
        assert_eq!(frame.get_trit(2), Trit::Zero);
        assert_eq!(frame.get_trit(9), Trit::Pos);
    }

    #[test]
    fn test_get_set_quaternary() {
        let mut frame = TernaryFrame::new(10, PackingMode::Quaternary);
        frame.set_trit(0, Trit::Pos);
        frame.set_trit(1, Trit::Neg);
        frame.set_trit(5, Trit::Pos);

        assert_eq!(frame.get_trit(0), Trit::Pos);
        assert_eq!(frame.get_trit(1), Trit::Neg);
        assert_eq!(frame.get_trit(2), Trit::Zero);
        assert_eq!(frame.get_trit(5), Trit::Pos);
    }

    #[test]
    fn test_fill() {
        let mut frame = TernaryFrame::new_ternary(20);
        frame.fill(5, 10, Trit::Pos);

        for i in 0..5 {
            assert_eq!(frame.get_trit(i), Trit::Zero);
        }
        for i in 5..15 {
            assert_eq!(frame.get_trit(i), Trit::Pos, "trit {} should be +1", i);
        }
        for i in 15..20 {
            assert_eq!(frame.get_trit(i), Trit::Zero);
        }
    }

    #[test]
    fn test_to_from_words() {
        let mut frame = TernaryFrame::new_ternary(128);
        frame.set_trit(0, Trit::Pos);
        frame.set_trit(63, Trit::Neg);
        frame.set_trit(64, Trit::Pos);
        frame.set_trit(127, Trit::Neg);

        let words = frame.to_words(0, 2);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].get_trit(0), 1);
        assert_eq!(words[0].get_trit(63), -1);
        assert_eq!(words[1].get_trit(0), 1);
        assert_eq!(words[1].get_trit(63), -1);

        // Write back
        let mut frame2 = TernaryFrame::new_ternary(128);
        frame2.from_words(&words, 0);
        assert!(frame.eq_frame(&frame2));
    }

    #[test]
    fn test_copy_from() {
        let mut src = TernaryFrame::new_ternary(10);
        src.set_trit(0, Trit::Pos);
        src.set_trit(1, Trit::Neg);

        let mut dst = TernaryFrame::new_ternary(10);
        let copied = dst.copy_from(3, &src, 0, 2);
        assert_eq!(copied, 2);
        assert_eq!(dst.get_trit(3), Trit::Pos);
        assert_eq!(dst.get_trit(4), Trit::Neg);
    }

    #[test]
    fn test_popcount() {
        let mut frame = TernaryFrame::new_ternary(10);
        assert_eq!(frame.popcount(), 0);

        frame.set_trit(0, Trit::Pos);
        frame.set_trit(5, Trit::Neg);
        assert_eq!(frame.popcount(), 2);
    }

    #[test]
    fn test_resize() {
        let mut frame = TernaryFrame::new_ternary(10);
        frame.set_trit(0, Trit::Pos);
        frame.set_trit(9, Trit::Neg);

        frame.resize(20);
        assert_eq!(frame.trit_count(), 20);
        assert_eq!(frame.get_trit(0), Trit::Pos);
        assert_eq!(frame.get_trit(9), Trit::Neg);

        frame.resize(5);
        assert_eq!(frame.trit_count(), 5);
        assert_eq!(frame.get_trit(0), Trit::Pos);
    }

    #[test]
    fn test_equality() {
        let mut a = TernaryFrame::new_ternary(10);
        let mut b = TernaryFrame::new_ternary(10);
        assert_eq!(a, b);

        a.set_trit(3, Trit::Pos);
        assert_ne!(a, b);

        b.set_trit(3, Trit::Pos);
        assert_eq!(a, b);
    }

    #[test]
    fn test_display() {
        let mut frame = TernaryFrame::new_ternary(3);
        frame.set_trit(0, Trit::Pos);
        frame.set_trit(1, Trit::Neg);
        let s = format!("{}", frame);
        assert!(s.contains("+1"));
        assert!(s.contains("-1"));
        assert!(s.contains("3 trits"));
    }
}
