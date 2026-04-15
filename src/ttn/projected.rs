//! Projected Tree Tensor Network — million-qubit scaling via
//! commensurability partitioning.
//!
//! [`ProjectedTtn`] is the composition of Track D's TTN simulator with
//! Track F's partitioner: it owns multiple [`VolatileIsland`] TTNs (one
//! per connected component of volatile edges) plus analytical boundary
//! tensors and stable-qubit observables, and presents a **unified
//! observable API over the full graph** — potentially millions of qubits.
//!
//! Track F milestone **F.4**.
//!
//! # How it works
//!
//! ```text
//! Full graph (1M qubits)
//!   → partition_tree_adaptive → Stable / Volatile edges
//!   → extract_volatile_islands → N small TTNs (~100s of qubits each)
//!   → compute_all_boundary_tensors → fixed boundary conditions
//!   → per-island Ttn::new + gate evolution
//!   → expectation_z_all → merge analytical + TTN observables
//!   → single Vec<f64> over all 1M qubits
//! ```
//!
//! The key scaling insight: only the volatile islands get site tensors
//! and SVDs. Stable qubits are handled analytically (their dynamics are
//! integrable by the commensurability argument). The partition determines
//! the simulation domain; the TTN handles that domain; the boundary
//! tensors glue the two together.

use crate::error::Result;
use crate::kicked_ising::KimParams;
use crate::ttn::boundary::{
    compute_all_boundary_tensors, BoundaryMode, BoundaryTensor,
};
use crate::ttn::feedback::{self, IslandFeedback, SolveFeedback};
use crate::ttn::kim_heavy_hex::apply_kim_step_heavy_hex_per_edge;
use crate::ttn::partition::TreePartition;
use crate::ttn::subtree::{extract_volatile_islands, VolatileIsland};
use crate::ttn::topology::{Edge, Topology};
use crate::ttn::Ttn;

/// State of one volatile island.
struct IslandState {
    ttn: Ttn,
    island: VolatileIsland,
    boundary_tensors: Vec<BoundaryTensor>,
}

/// Projected Tree Tensor Network over a full coupling graph.
///
/// Owns multiple volatile-island TTNs + analytical observables for stable
/// qubits. The public API presents a unified `Vec<f64>` of per-qubit ⟨Z⟩
/// over the entire graph.
pub struct ProjectedTtn {
    /// Full-graph qubit count.
    n_qubits_total: usize,
    /// The partition that determined stable / volatile classification.
    partition: TreePartition,
    /// Per-island TTN state.
    islands: Vec<IslandState>,
    /// Pre-computed analytical ⟨Z⟩ for stable qubits (indexed by global
    /// qubit id; volatile qubits have 0.0 here — they're filled from TTN).
    stable_z: Vec<f64>,
    /// Non-tree coupling edges for each island (in **local** indices),
    /// pre-computed at construction. Empty if the full graph has no
    /// non-tree edges, or if a non-tree edge falls entirely in the stable
    /// region (in which case it's skipped).
    island_non_tree_edges: Vec<Vec<Edge>>,
}

impl ProjectedTtn {
    /// Build a projected TTN from a full topology + partition.
    ///
    /// # Arguments
    ///
    /// * `full_topology` — the full spanning tree (may be lightweight).
    /// * `frequencies` — per-qubit frequency vector (length ≥ n_qubits).
    /// * `partition` — the stable/volatile classification from
    ///   [`crate::ttn::partition::partition_tree_adaptive`].
    /// * `non_tree_edges` — coupling-graph edges not in the spanning tree
    ///   (e.g. from [`HeavyHexLayout::non_tree_edges`]).
    /// * `params` — KIM parameters for boundary-tensor computation.
    /// * `n_boundary_steps` — number of Floquet steps to pre-evolve the
    ///   boundary tensors (typically the same as the circuit depth).
    /// * `mode` — boundary-tensor computation mode.
    #[must_use]
    pub fn new(
        full_topology: &Topology,
        frequencies: &[f64],
        partition: TreePartition,
        non_tree_edges_global: &[[usize; 2]],
        params: KimParams,
        n_boundary_steps: usize,
        mode: BoundaryMode,
    ) -> Self {
        let n = full_topology.n_qubits();

        // 1. Extract volatile islands.
        let islands = extract_volatile_islands(full_topology, &partition);

        // 2. Compute boundary tensors for each island.
        let all_bt = compute_all_boundary_tensors(
            frequencies,
            &islands,
            params,
            n_boundary_steps,
            mode,
        );

        // 3. Build a Ttn for each island.
        let mut island_states: Vec<IslandState> = islands
            .into_iter()
            .zip(all_bt)
            .map(|(island, bt)| {
                let ttn = Ttn::new(island.topology.clone());
                IslandState {
                    ttn,
                    island,
                    boundary_tensors: bt,
                }
            })
            .collect();

        // 4. Map non-tree edges to island-local indices. A non-tree edge
        //    (a, b) is dispatched to the island containing BOTH a and b.
        //    If either endpoint is stable, the edge is skipped (its gate
        //    dynamics are handled analytically).
        let mut island_non_tree: Vec<Vec<Edge>> = vec![Vec::new(); island_states.len()];
        for &[ga, gb] in non_tree_edges_global {
            // Find the island containing both endpoints.
            for (idx, is) in island_states.iter().enumerate() {
                if let (Some(la), Some(lb)) = (
                    is.island.global_to_local.get(ga).and_then(|x| *x),
                    is.island.global_to_local.get(gb).and_then(|x| *x),
                ) {
                    island_non_tree[idx].push(Edge { a: la, b: lb });
                    break;
                }
            }
            // If neither island claims both endpoints, the edge spans
            // the stable region — skipped.
        }

        // 5. Pre-compute stable ⟨Z⟩. For ProductState boundaries, each
        //    stable qubit evolves under single-qubit Rx / Rz kicks and
        //    its ⟨Z⟩ is deterministic. We compute it via a boundary
        //    tensor at the qubit itself (which is slightly over-general
        //    but reuses the existing code).
        let mut stable_z = vec![0.0_f64; n];
        for &q in &partition.stable_qubits {
            let omega = frequencies.get(q).copied().unwrap_or(params.h_x);
            let bt = crate::ttn::boundary::compute_boundary_tensor(
                omega, params, n_boundary_steps, mode,
            );
            stable_z[q] = bt.expectation_z();
        }

        // 6. Also fill stable_z for boundary qubits on the STABLE side
        //    (they're in partition.stable_qubits if they only touch stable
        //    edges; otherwise they're volatile_qubits and handled by TTN).

        ProjectedTtn {
            n_qubits_total: n,
            partition,
            islands: island_states,
            stable_z,
            island_non_tree_edges: island_non_tree,
        }
    }

    /// Apply one KIM Floquet step to every volatile island's TTN.
    ///
    /// Gates within stable regions are **not applied** — their dynamics
    /// are captured by the boundary tensors pre-computed at construction.
    /// Gates within volatile islands are dispatched to the island's TTN
    /// via [`apply_kim_step_heavy_hex_per_edge`]. Gates crossing the
    /// stable/volatile boundary are handled by the boundary tensor
    /// (not yet implemented in this milestone — stubbed as a skip).
    pub fn apply_floquet_step(&mut self, params: KimParams) -> Result<()> {
        for (idx, is) in self.islands.iter_mut().enumerate() {
            let non_tree = &self.island_non_tree_edges[idx];
            // Use chi_per_edge from the partition, with a sensible cap
            // for the non-tree swap-network path.
            let chi_max = is.island.chi_per_edge.iter().copied().max().unwrap_or(8);
            apply_kim_step_heavy_hex_per_edge(
                &mut is.ttn,
                non_tree,
                params,
                &is.island.chi_per_edge,
                chi_max,
            )?;
        }
        Ok(())
    }

    /// Return ⟨Z⟩ for **every** qubit in the full graph.
    ///
    /// - Stable qubits: analytical value computed at construction.
    /// - Volatile qubits: from the island TTN's `expectation_z_all`.
    /// - Boundary qubits (volatile side): from the island TTN.
    /// - Boundary qubits (stable side): from the analytical model.
    pub fn expectation_z_all(&mut self) -> Vec<f64> {
        let mut z = self.stable_z.clone();
        for is in &mut self.islands {
            let island_z = is.ttn.expectation_z_all();
            for (local, &global) in is.island.local_to_global.iter().enumerate() {
                z[global] = island_z[local];
            }
        }
        z
    }

    /// Total qubit count in the full graph.
    #[must_use]
    pub fn n_qubits_total(&self) -> usize {
        self.n_qubits_total
    }

    /// Number of qubits actually simulated via TTN (across all islands).
    #[must_use]
    pub fn n_qubits_volatile(&self) -> usize {
        self.islands
            .iter()
            .map(|is| is.ttn.n_qubits())
            .sum()
    }

    /// Number of volatile islands.
    #[must_use]
    pub fn n_islands(&self) -> usize {
        self.islands.len()
    }

    /// Total cumulative discarded weight across all island TTNs.
    #[must_use]
    pub fn total_discarded_weight(&self) -> f64 {
        self.islands
            .iter()
            .map(|is| is.ttn.total_discarded_weight())
            .sum()
    }

    /// Collect per-island solve statistics for Arn's feedback loop.
    ///
    /// Call this after one or more `apply_floquet_step` calls. The
    /// `solve_ms` field is left as `None` — the caller should fill it
    /// from their own timing if they want wall-time feedback.
    #[must_use]
    pub fn collect_feedback(&self, frequencies: &[f64]) -> SolveFeedback {
        let n_vol: usize = self.islands.iter().map(|is| is.ttn.n_qubits()).sum();
        let islands: Vec<IslandFeedback> = self.islands.iter().map(|is| {
            let n_q = is.ttn.n_qubits();
            let n_e = is.island.topology.n_edges();
            let max_deg = (0..n_q)
                .map(|v| is.island.topology.degree(v))
                .max()
                .unwrap_or(0);
            let chi = is.island.chi_per_edge.clone();
            let total_chi: usize = chi.iter().sum();
            let dw = is.ttn.total_discarded_weight();
            let freqs: Vec<f64> = is.island.local_to_global.iter()
                .map(|&g| frequencies.get(g).copied().unwrap_or(0.0))
                .collect();
            IslandFeedback {
                n_qubits: n_q,
                n_edges: n_e,
                max_degree: max_deg,
                chi_per_edge: chi,
                total_chi,
                discarded_weight: dw,
                n_boundary_edges: is.island.boundary_edges.len(),
                frequencies: freqs,
            }
        }).collect();

        let total_dw: f64 = islands.iter().map(|i| i.discarded_weight).sum();

        SolveFeedback {
            n_qubits_total: self.n_qubits_total,
            n_islands: self.islands.len(),
            n_qubits_volatile: n_vol,
            volatile_fraction: if self.n_qubits_total > 0 {
                n_vol as f64 / self.n_qubits_total as f64
            } else {
                0.0
            },
            islands,
            total_discarded_weight: total_dw,
            solve_ms: None,
        }
    }

    /// Collect feedback and POST it to Arn's `/feedback` endpoint.
    ///
    /// Best-effort: if Arn is unreachable, the call is silently dropped.
    /// The simulation result is never affected by feedback delivery.
    pub fn post_feedback(&self, frequencies: &[f64], arn_url: &str) {
        let fb = self.collect_feedback(frequencies);
        feedback::post_feedback(arn_url, &fb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttn::partition::partition_tree_adaptive;

    #[test]
    fn all_stable_returns_analytical_for_every_qubit() {
        let topology = Topology::linear_chain(10);
        let frequencies = vec![1.0; 10]; // uniform → all stable
        let partition = partition_tree_adaptive(&frequencies, &topology, 72, 2, 16, 5);
        assert!(partition.volatile_edges.is_empty());

        let params = KimParams::self_dual();
        let mut pttn = ProjectedTtn::new(
            &topology, &frequencies, partition, &[], params, 5, BoundaryMode::ProductState,
        );
        assert_eq!(pttn.n_qubits_total(), 10);
        assert_eq!(pttn.n_qubits_volatile(), 0);
        assert_eq!(pttn.n_islands(), 0);

        let z = pttn.expectation_z_all();
        assert_eq!(z.len(), 10);
        for &zq in &z {
            assert!(zq.is_finite() && (-1.0..=1.0).contains(&zq));
        }
    }

    #[test]
    fn all_volatile_behaves_like_plain_ttn() {
        let topology = Topology::linear_chain(6);
        let frequencies = [
            1.0,
            2.0_f64.sqrt(),
            3.0_f64.sqrt(),
            5.0_f64.sqrt(),
            7.0_f64.sqrt(),
            11.0_f64.sqrt(),
        ];
        let partition = partition_tree_adaptive(&frequencies, &topology, 100, 2, 32, 5);
        assert_eq!(partition.volatile_edges.len(), 5);

        let params = KimParams::self_dual();
        let mut pttn = ProjectedTtn::new(
            &topology,
            &frequencies,
            partition,
            &[],
            params,
            0,
            BoundaryMode::ProductState,
        );
        assert_eq!(pttn.n_qubits_total(), 6);
        assert_eq!(pttn.n_qubits_volatile(), 6);
        assert_eq!(pttn.n_islands(), 1);

        // Initial state: all qubits in |0⟩ → ⟨Z⟩ = +1.
        let z = pttn.expectation_z_all();
        for &zq in &z {
            assert!((zq - 1.0).abs() < 1e-12);
        }

        // Apply one step and check it's non-trivial.
        pttn.apply_floquet_step(params).unwrap();
        let z2 = pttn.expectation_z_all();
        let any_moved = z2.iter().any(|&zq| (zq - 1.0).abs() > 0.01);
        assert!(any_moved, "at least one qubit should have moved after a Floquet step");
    }

    #[test]
    fn eagle_127q_projected_with_disorder() {
        use crate::ttn::heavy_hex::HeavyHexLayout;

        let layout = HeavyHexLayout::ibm_eagle_127();
        let primes = [2.0_f64, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0];
        let frequencies: Vec<f64> = (0..127)
            .map(|i| 0.8 + 0.05 * primes[i % primes.len()].sqrt())
            .collect();

        let partition = partition_tree_adaptive(
            &frequencies,
            layout.tree(),
            504,
            2,
            32,
            5,
        );
        let n_volatile = partition.volatile_qubits.len();
        assert!(n_volatile > 0 && n_volatile < 127);

        let params = KimParams {
            j: -std::f64::consts::FRAC_PI_4,
            h_x: 0.4,
            h_z: 0.0,
            dt: 1.0,
        };

        let mut pttn = ProjectedTtn::new(
            layout.tree(),
            &frequencies,
            partition,
            layout.non_tree_edges(),
            params,
            3,
            BoundaryMode::ProductState,
        );

        assert_eq!(pttn.n_qubits_total(), 127);
        assert!(pttn.n_qubits_volatile() > 0);
        assert!(pttn.n_qubits_volatile() < 127);

        // Apply 3 Floquet steps.
        for _ in 0..3 {
            pttn.apply_floquet_step(params).unwrap();
        }

        let z = pttn.expectation_z_all();
        assert_eq!(z.len(), 127);
        for (q, &zq) in z.iter().enumerate() {
            assert!(
                zq.is_finite() && (-1.0..=1.0).contains(&zq),
                "q={q}: ⟨Z⟩ = {zq} invalid"
            );
        }

        // Discarded weight should be finite.
        let disc = pttn.total_discarded_weight();
        assert!(disc.is_finite() && disc >= 0.0);

        eprintln!(
            "[F.4] Eagle 127q projected: {} volatile qubits in {} islands, \
             discarded weight = {disc:.4e}",
            pttn.n_qubits_volatile(),
            pttn.n_islands()
        );
    }

    #[test]
    fn observable_vector_has_correct_length() {
        let topology = Topology::linear_chain(20);
        let frequencies: Vec<f64> = (0..20)
            .map(|i| 1.0 + (i as f64).sqrt() * 0.2)
            .collect();
        let partition = partition_tree_adaptive(&frequencies, &topology, 100, 2, 16, 5);
        let params = KimParams::self_dual();
        let mut pttn = ProjectedTtn::new(
            &topology, &frequencies, partition, &[], params, 0, BoundaryMode::ProductState,
        );
        let z = pttn.expectation_z_all();
        assert_eq!(z.len(), 20);
        // Every value should be finite and bounded.
        for &zq in &z {
            assert!(zq.is_finite() && (-1.0..=1.0).contains(&zq));
        }
    }

    #[test]
    fn collect_feedback_has_correct_shape() {
        let topology = Topology::linear_chain(20);
        let frequencies: Vec<f64> = (0..20)
            .map(|i| if i % 3 == 0 { (i as f64 + 2.0).sqrt() } else { 1.0 })
            .collect();
        let partition = partition_tree_adaptive(&frequencies, &topology, 100, 2, 16, 5);
        let params = KimParams::self_dual();
        let mut pttn = ProjectedTtn::new(
            &topology, &frequencies, partition, &[], params, 0, BoundaryMode::ProductState,
        );

        // Run a step so there's discarded weight to report.
        pttn.apply_floquet_step(params).unwrap();

        let fb = pttn.collect_feedback(&frequencies);

        assert_eq!(fb.n_qubits_total, 20);
        assert_eq!(fb.n_islands, pttn.n_islands());
        assert_eq!(fb.n_qubits_volatile, pttn.n_qubits_volatile());
        assert!(fb.volatile_fraction >= 0.0 && fb.volatile_fraction <= 1.0);
        assert!(fb.total_discarded_weight >= 0.0);
        assert!(fb.solve_ms.is_none()); // caller fills this

        // Per-island checks.
        for island_fb in &fb.islands {
            assert!(island_fb.n_qubits > 0);
            assert_eq!(island_fb.n_edges, island_fb.n_qubits - 1); // tree
            assert_eq!(island_fb.chi_per_edge.len(), island_fb.n_edges);
            assert_eq!(island_fb.total_chi, island_fb.chi_per_edge.iter().sum::<usize>());
            assert_eq!(island_fb.frequencies.len(), island_fb.n_qubits);
            assert!(island_fb.discarded_weight >= 0.0);
        }

        // Serializes to valid JSON.
        let json = serde_json::to_string(&fb).unwrap();
        assert!(json.contains("\"n_qubits_total\":20"));
        assert!(json.contains("\"volatile_fraction\":"));

        eprintln!(
            "[feedback] {} islands, {} volatile qubits, dw={:.4e}, json={} bytes",
            fb.n_islands, fb.n_qubits_volatile, fb.total_discarded_weight, json.len()
        );
    }
}
