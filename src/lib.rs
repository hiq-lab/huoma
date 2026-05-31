//! `huoma` — Projector simulator with commensurability-guided MPS.
//!
//! Decomposes a quantum circuit into stable (analytically solvable) and
//! volatile (must-simulate) partitions using the sin(C/2) commensurability
//! filter from the Tilde Pattern (Hinderink 2026). Only the volatile
//! partition is simulated via MPS with adaptive per-bond truncation.
//!
//! # Architecture
//!
//! ```text
//! Circuit → frequency extraction → channel assessment → partitioning
//!     → stable (analytical) + volatile (MPS/QPU) → reassembly
//! ```
//!
//! # References
//!
//! - Hinderink (2026), "Pair-Counting Scaling of the Quantum Chaos Threshold"
//! - Hinderink (2026), "The Tilde Pattern: Commensurability as Low-Pass Filter"

pub mod allocator;
pub mod bianchi;
pub mod channel;
pub mod closed_surface;
pub mod error;
pub mod frequency;
pub mod hyperbolic;
pub mod kicked_ising;
pub mod magnetic;
pub mod mps;
pub mod partition;
pub mod peierls;
pub mod reassembly;
pub mod ttn;
pub mod xxz;

// Production-recommended χ allocators, re-exported at crate root.
pub use allocator::{
    chi_allocation_sinc, chi_allocation_sinc_with_radius, chi_allocation_target_budget,
};

#[cfg(test)]
mod accuracy;
#[cfg(test)]
mod bench;
#[cfg(test)]
mod shootout;
#[cfg(test)]
mod validation;
