//! Learning rate schedulers.
//!
//! Provides cosine annealing, warmup-cosine, and one-cycle schedulers.

use std::f32::consts::PI;

/// Trait for learning rate schedulers.
pub trait LrScheduler {
    /// Get the learning rate for the current step.
    fn get_lr(&self, step: usize) -> f32;

    /// Total number of steps (if finite).
    fn total_steps(&self) -> usize;
}

/// Cosine annealing scheduler: lr decays from `lr_max` to `lr_min` over `total_steps`.
///
/// lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
pub struct CosineAnnealing {
    lr_max: f32,
    lr_min: f32,
    total: usize,
}

impl CosineAnnealing {
    pub fn new(lr_max: f32, lr_min: f32, total_steps: usize) -> Self {
        Self {
            lr_max,
            lr_min,
            total: total_steps,
        }
    }
}

impl LrScheduler for CosineAnnealing {
    fn get_lr(&self, step: usize) -> f32 {
        if step >= self.total {
            return self.lr_min;
        }
        let progress = step as f32 / self.total as f32;
        self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + (PI * progress).cos())
    }

    fn total_steps(&self) -> usize {
        self.total
    }
}

/// Warmup + cosine annealing scheduler.
///
/// Linear warmup from `lr_start` to `lr_max` over `warmup_steps`,
/// then cosine decay from `lr_max` to `lr_min` over the remaining steps.
pub struct WarmupCosine {
    lr_start: f32,
    lr_max: f32,
    lr_min: f32,
    warmup_steps: usize,
    total: usize,
}

impl WarmupCosine {
    pub fn new(
        lr_start: f32,
        lr_max: f32,
        lr_min: f32,
        warmup_steps: usize,
        total_steps: usize,
    ) -> Self {
        assert!(
            warmup_steps < total_steps,
            "warmup_steps must be < total_steps"
        );
        Self {
            lr_start,
            lr_max,
            lr_min,
            warmup_steps,
            total: total_steps,
        }
    }
}

impl LrScheduler for WarmupCosine {
    fn get_lr(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            let progress = step as f32 / self.warmup_steps as f32;
            self.lr_start + (self.lr_max - self.lr_start) * progress
        } else if step >= self.total {
            self.lr_min
        } else {
            // Cosine decay
            let decay_steps = self.total - self.warmup_steps;
            let decay_step = step - self.warmup_steps;
            let progress = decay_step as f32 / decay_steps as f32;
            self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + (PI * progress).cos())
        }
    }

    fn total_steps(&self) -> usize {
        self.total
    }
}

/// One-cycle learning rate scheduler (Smith, 2018).
///
/// Phase 1 (0..peak_step): linear warmup from `lr_min` to `lr_max`
/// Phase 2 (peak_step..total): cosine decay from `lr_max` to `lr_min / 10`
pub struct OneCycle {
    lr_max: f32,
    lr_min: f32,
    peak_step: usize,
    total: usize,
}

impl OneCycle {
    /// Create a one-cycle scheduler.
    /// `pct_start` is the fraction of total steps spent in the warmup phase (typically 0.3).
    pub fn new(lr_max: f32, total_steps: usize, pct_start: f32) -> Self {
        let peak_step = (total_steps as f32 * pct_start) as usize;
        Self {
            lr_max,
            lr_min: lr_max / 25.0, // typical initial lr
            peak_step,
            total: total_steps,
        }
    }

    /// Create with explicit min lr.
    pub fn with_min_lr(lr_max: f32, lr_min: f32, total_steps: usize, pct_start: f32) -> Self {
        let peak_step = (total_steps as f32 * pct_start) as usize;
        Self {
            lr_max,
            lr_min,
            peak_step,
            total: total_steps,
        }
    }
}

impl LrScheduler for OneCycle {
    fn get_lr(&self, step: usize) -> f32 {
        if step >= self.total {
            return self.lr_min / 10.0;
        }

        if step <= self.peak_step {
            // Phase 1: linear warmup
            if self.peak_step == 0 {
                return self.lr_max;
            }
            let progress = step as f32 / self.peak_step as f32;
            self.lr_min + (self.lr_max - self.lr_min) * progress
        } else {
            // Phase 2: cosine decay to lr_min / 10
            let decay_steps = self.total - self.peak_step;
            let decay_step = step - self.peak_step;
            let progress = decay_step as f32 / decay_steps as f32;
            let final_lr = self.lr_min / 10.0;
            final_lr + 0.5 * (self.lr_max - final_lr) * (1.0 + (PI * progress).cos())
        }
    }

    fn total_steps(&self) -> usize {
        self.total
    }
}

/// Step decay scheduler: lr = lr_init * gamma^(step / step_size)
pub struct StepDecay {
    lr_init: f32,
    gamma: f32,
    step_size: usize,
    total: usize,
}

impl StepDecay {
    pub fn new(lr_init: f32, gamma: f32, step_size: usize, total_steps: usize) -> Self {
        Self {
            lr_init,
            gamma,
            step_size,
            total: total_steps,
        }
    }
}

impl LrScheduler for StepDecay {
    fn get_lr(&self, step: usize) -> f32 {
        let n = step / self.step_size;
        self.lr_init * self.gamma.powi(n as i32)
    }

    fn total_steps(&self) -> usize {
        self.total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_annealing() {
        let sched = CosineAnnealing::new(0.01, 0.0, 100);

        // Start: should be lr_max
        assert!((sched.get_lr(0) - 0.01).abs() < 1e-6);

        // Middle: should be ~lr_max/2
        let mid = sched.get_lr(50);
        assert!((mid - 0.005).abs() < 1e-4, "mid={}", mid);

        // End: should be lr_min
        assert!((sched.get_lr(100) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_monotonic_decrease() {
        let sched = CosineAnnealing::new(0.1, 0.0, 1000);
        let mut prev = sched.get_lr(0);
        for step in 1..=1000 {
            let lr = sched.get_lr(step);
            assert!(lr <= prev + 1e-7, "step {}: {} > {}", step, lr, prev);
            prev = lr;
        }
    }

    #[test]
    fn test_warmup_cosine() {
        let sched = WarmupCosine::new(0.0, 0.01, 0.0, 100, 1000);

        // Start: lr_start
        assert!((sched.get_lr(0) - 0.0).abs() < 1e-6);

        // End of warmup: lr_max
        let warmup_end = sched.get_lr(100);
        assert!((warmup_end - 0.01).abs() < 1e-4, "warmup_end={}", warmup_end);

        // After warmup, should decrease
        let after = sched.get_lr(500);
        assert!(after < 0.01);

        // End: lr_min
        assert!((sched.get_lr(1000) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_warmup_increasing() {
        let sched = WarmupCosine::new(0.0, 0.01, 0.0, 100, 1000);
        let mut prev = sched.get_lr(0);
        for step in 1..100 {
            let lr = sched.get_lr(step);
            assert!(lr >= prev - 1e-7, "warmup step {}: {} < {}", step, lr, prev);
            prev = lr;
        }
    }

    #[test]
    fn test_one_cycle() {
        let sched = OneCycle::new(0.01, 1000, 0.3);

        // Start: should be lr_min (lr_max/25)
        let start = sched.get_lr(0);
        assert!((start - 0.0004).abs() < 1e-4, "start={}", start);

        // Peak: should be lr_max
        let peak = sched.get_lr(300);
        assert!((peak - 0.01).abs() < 1e-4, "peak={}", peak);

        // End: should be very small
        let end = sched.get_lr(1000);
        assert!(end < 0.001, "end={}", end);
    }

    #[test]
    fn test_step_decay() {
        let sched = StepDecay::new(0.1, 0.5, 100, 500);

        assert!((sched.get_lr(0) - 0.1).abs() < 1e-6);
        assert!((sched.get_lr(99) - 0.1).abs() < 1e-6);
        assert!((sched.get_lr(100) - 0.05).abs() < 1e-6);
        assert!((sched.get_lr(200) - 0.025).abs() < 1e-6);
    }
}
