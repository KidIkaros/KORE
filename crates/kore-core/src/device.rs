use std::fmt;

/// Compute device for tensor storage and operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// CPU with optional SIMD acceleration (AVX2, AVX-512, NEON)
    #[default]
    Cpu,
    /// CUDA GPU with device index
    Cuda(usize),
}

impl Device {
    /// Whether this is a CPU device.
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Whether this is a CUDA device.
    pub fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }

    /// Get the CUDA device index, if applicable.
    pub fn cuda_index(&self) -> Option<usize> {
        match self {
            Device::Cuda(idx) => Some(*idx),
            _ => None,
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(idx) => write!(f, "cuda:{idx}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_properties() {
        assert!(Device::Cpu.is_cpu());
        assert!(!Device::Cpu.is_cuda());
        assert!(Device::Cuda(0).is_cuda());
        assert_eq!(Device::Cuda(1).cuda_index(), Some(1));
        assert_eq!(Device::Cpu.cuda_index(), None);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Device::Cpu), "cpu");
        assert_eq!(format!("{}", Device::Cuda(0)), "cuda:0");
    }

    #[test]
    fn test_default() {
        assert_eq!(Device::default(), Device::Cpu);
    }
}
