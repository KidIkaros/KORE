//! Gradient computation scopes.

use std::cell::Cell;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Check if gradient computation is currently enabled.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|g| g.get())
}

/// Set whether gradient computation is enabled.
fn set_grad_enabled(enabled: bool) -> bool {
    GRAD_ENABLED.with(|g| {
        let prev = g.get();
        g.set(enabled);
        prev
    })
}

/// RAII guard that disables gradient computation in its scope.
///
/// # Example
/// ```
/// use kore_autograd::NoGradGuard;
///
/// {
///     let _guard = NoGradGuard::new();
///     // All operations here skip gradient tracking
/// }
/// // Gradients re-enabled when guard is dropped
/// ```
pub struct NoGradGuard {
    prev: bool,
}

impl NoGradGuard {
    /// Create a new no-grad scope.
    pub fn new() -> Self {
        let prev = set_grad_enabled(false);
        Self { prev }
    }
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        set_grad_enabled(self.prev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_grad_guard() {
        assert!(is_grad_enabled());

        {
            let _guard = NoGradGuard::new();
            assert!(!is_grad_enabled());

            {
                let _inner = NoGradGuard::new();
                assert!(!is_grad_enabled());
            }
            // Inner guard dropped, still disabled because outer guard
            assert!(!is_grad_enabled());
        }

        // Outer guard dropped, restored to true
        assert!(is_grad_enabled());
    }
}
