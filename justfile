# Justfile for KORE ML Engine
# https://github.com/casey/just

# Default recipe - show available commands
default:
    @just --list

# Run all tests
test:
    cargo test --workspace

# Run tests for a specific crate
test-crate crate:
    cargo test -p {{crate}}

# Run clippy linting on all packages
lint:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

# Auto-fix clippy warnings where possible
lint-fix:
    cargo clippy --workspace --all-targets --all-features --fix --allow-dirty

# Format all code
fmt:
    cargo fmt --all

# Check formatting without modifying
fmt-check:
    cargo fmt --all -- --check

# Build documentation
docs:
    cargo doc --workspace --no-deps

# Build and serve documentation locally
docs-serve:
    cargo doc --workspace --no-deps --open

# Run benchmarks
bench:
    cargo bench --workspace

# Run benchmarks for a specific crate
bench-crate crate:
    cargo bench -p {{crate}}

# Build release binaries
build:
    cargo build --workspace --release

# Clean build artifacts
clean:
    cargo clean

# Check all packages (fast compilation check)
check:
    cargo check --workspace --all-features

# Run security audit on dependencies
audit:
    cargo audit

# Update dependencies
update:
    cargo update

# Check for outdated dependencies
outdated:
    cargo outdated

# Run all quality checks (lint + test + fmt-check)
check-all: fmt-check lint test
    @echo "✅ All quality checks passed!"

# Quick development cycle - check and test
dev: check test
    @echo "✅ Development checks complete!"

# Full CI simulation locally
ci: clean fmt-check lint test
    @echo "✅ CI simulation passed!"
