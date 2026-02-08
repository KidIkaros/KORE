//! # kore-clifford
//!
//! Geometric algebra engine for Kore.
//!
//! Provides Clifford algebras Cl(p,q) with:
//! - Compile-time Cayley table generation
//! - Multivector storage and arithmetic
//! - Geometric, inner, outer products
//! - Grade projection, dual, reverse, involute
//! - Norm and normalization

pub mod algebra;
pub mod multivector;
pub mod products;
pub mod ops;

pub use algebra::CliffordAlgebra;
pub use multivector::Multivector;
