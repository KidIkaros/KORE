use std::fmt;

/// Compute device for tensor storage and operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// CPU with optional SIMD acceleration (AVX2, AVX-512, NEON)
    #[default]
    Cpu,
    /// CUDA GPU with device index
    Cuda(usize),
    /// Vulkan GPU with device index
    Vulkan(usize),
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

    /// Whether this is a Vulkan device.
    pub fn is_vulkan(&self) -> bool {
        matches!(self, Device::Vulkan(_))
    }

    /// Get the Vulkan device index, if applicable.
    pub fn vulkan_index(&self) -> Option<usize> {
        match self {
            Device::Vulkan(idx) => Some(*idx),
            _ => None,
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(idx) => write!(f, "cuda:{idx}"),
            Device::Vulkan(idx) => write!(f, "vulkan:{idx}"),
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

        assert!(Device::Vulkan(0).is_vulkan());
        assert!(!Device::Cpu.is_vulkan());
        assert_eq!(Device::Vulkan(2).vulkan_index(), Some(2));
        assert_eq!(Device::Cpu.vulkan_index(), None);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Device::Cpu), "cpu");
        assert_eq!(format!("{}", Device::Cuda(0)), "cuda:0");
        assert_eq!(format!("{}", Device::Vulkan(0)), "vulkan:0");
    }

    #[test]
    fn test_default() {
        assert_eq!(Device::default(), Device::Cpu);
    }
}
