//! Runtime SIMD capability detection.
//!
//! Detects AVX2, AVX-512, and ARM NEON at runtime.
//! Ported from btes/src/qbtes_matmul_avx2.c detect_cpu_features().

use std::sync::OnceLock;

/// SIMD capabilities detected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimdCapability {
    pub avx2: bool,
    pub avx512f: bool,
    pub neon: bool,
    pub fma: bool,
}

static DETECTED: OnceLock<SimdCapability> = OnceLock::new();

impl SimdCapability {
    /// Detect SIMD capabilities for the current CPU.
    pub fn detect() -> &'static SimdCapability {
        DETECTED.get_or_init(|| {
            #[cfg(target_arch = "x86_64")]
            {
                SimdCapability {
                    avx2: is_x86_feature_detected!("avx2"),
                    avx512f: is_x86_feature_detected!("avx512f"),
                    fma: is_x86_feature_detected!("fma"),
                    neon: false,
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                SimdCapability {
                    avx2: false,
                    avx512f: false,
                    fma: false,
                    neon: true, // NEON is mandatory on AArch64
                }
            }

            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                SimdCapability {
                    avx2: false,
                    avx512f: false,
                    fma: false,
                    neon: false,
                }
            }
        })
    }

    /// Best available SIMD tier as a human-readable string.
    pub fn best_tier(&self) -> &'static str {
        if self.avx512f {
            "AVX-512"
        } else if self.avx2 {
            "AVX2"
        } else if self.neon {
            "NEON"
        } else {
            "scalar"
        }
    }

    /// Whether any SIMD is available.
    pub fn has_simd(&self) -> bool {
        self.avx2 || self.avx512f || self.neon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect() {
        let cap = SimdCapability::detect();
        // Just verify it doesn't panic and returns consistent results
        let cap2 = SimdCapability::detect();
        assert_eq!(cap, cap2);
        println!("SIMD: {} (avx2={}, avx512={}, neon={}, fma={})",
            cap.best_tier(), cap.avx2, cap.avx512f, cap.neon, cap.fma);
    }

    #[test]
    fn test_best_tier() {
        let scalar = SimdCapability { avx2: false, avx512f: false, neon: false, fma: false };
        assert_eq!(scalar.best_tier(), "scalar");
        assert!(!scalar.has_simd());

        let avx2 = SimdCapability { avx2: true, avx512f: false, neon: false, fma: true };
        assert_eq!(avx2.best_tier(), "AVX2");
        assert!(avx2.has_simd());
    }
}
