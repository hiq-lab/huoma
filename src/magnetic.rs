//! Magnetic-field tight-binding Hamiltonians via Peierls substitution,
//! in the **spectral** mode: build the static matrix, diagonalise, sweep
//! flux. Complementary to the time-evolution path in `src/peierls.rs`
//! (which provides the same physics as TEBD gates for closed-system
//! unitary evolution).
//!
//! The headline application is the **Hofstadter butterfly** —
//! eigenvalue spectrum of a tight-binding model on a 2D lattice as a
//! function of magnetic flux per plaquette, swept across `Φ/Φ₀ ∈ [0, 1]`.
//! Square-lattice butterfly (Hofstadter 1976, Phys. Rev. B 14, 2239) is
//! the canonical validation anchor. Hyperbolic-lattice butterflies
//! (Stegmaier et al. 2022; Lenggenhager et al., Nat. Commun. 13, 2022)
//! are the foundational new physics output Track F existed for.
//!
//! # Gauge convention
//!
//! Landau gauge on the embedding chart: for an oriented edge `i → j`
//! with positions `(x_i, y_i), (x_j, y_j)`, the Peierls phase is
//!
//! ```text
//! φ_ij = -2π · Φ · y_avg · Δx,    y_avg = (y_i + y_j)/2,  Δx = x_j - x_i
//! ```
//!
//! Stokes' theorem gives `Σ φ around plaquette = 2π · Φ · (area enclosed)`
//! in the embedding plane. For square lattice with unit spacing this is
//! `2π Φ` exactly per plaquette — the textbook Hofstadter normalisation.
//!
//! For a hyperbolic tiling embedded in the Poincaré disk, this is the
//! "uniform magnetic field on the embedding chart" (i.e. uniform in
//! Euclidean disk coords, not in the hyperbolic area form). It is the
//! standard first-order convention in the hyperbolic-Hofstadter
//! literature and is the simplest gauge that admits Stokes-theorem
//! verification. A future refinement could use the hyperbolic area
//! form `dx∧dy / (1 − x² − y²)²` for "uniform hyperbolic B field"
//! proper.

use faer::{Mat, Side, c64};
use num_complex::Complex64;

use crate::hyperbolic::HyperbolicLayout;

/// A planar-embedded graph: vertex positions `(x, y)` in some chart
/// (Euclidean plane for square / triangular lattices, Poincaré disk
/// coordinates for hyperbolic) plus an edge list.
#[derive(Clone, Debug)]
pub struct EmbeddedGraph {
    pub positions: Vec<(f64, f64)>,
    /// Undirected edges as `(u, v)` with `u < v`.
    pub edges: Vec<(usize, usize)>,
}

impl EmbeddedGraph {
    #[must_use]
    pub fn n_vertices(&self) -> usize {
        self.positions.len()
    }

    /// `M × N` square lattice with unit spacing. Vertices on integer
    /// coordinates `(x, y)` for `0 ≤ x < cols`, `0 ≤ y < rows`,
    /// indexed row-major (`q = y * cols + x`). Edges: horizontal
    /// `(x, y) — (x+1, y)` and vertical `(x, y) — (x, y+1)` when in
    /// range. No periodic boundary.
    ///
    /// This is the canonical lattice for the textbook Hofstadter
    /// butterfly; an `n × n` square at `n ≥ 8` produces a
    /// recognisable fractal.
    #[must_use]
    pub fn square_lattice(rows: usize, cols: usize) -> Self {
        assert!(rows >= 2 && cols >= 2, "square_lattice requires rows, cols ≥ 2");
        let mut positions = Vec::with_capacity(rows * cols);
        for y in 0..rows {
            for x in 0..cols {
                positions.push((x as f64, y as f64));
            }
        }
        let mut edges = Vec::new();
        let idx = |x: usize, y: usize| -> usize { y * cols + x };
        for y in 0..rows {
            for x in 0..cols {
                if x + 1 < cols {
                    edges.push((idx(x, y), idx(x + 1, y)));
                }
                if y + 1 < rows {
                    edges.push((idx(x, y), idx(x, y + 1)));
                }
            }
        }
        Self { positions, edges }
    }
}

impl From<&HyperbolicLayout> for EmbeddedGraph {
    fn from(layout: &HyperbolicLayout) -> Self {
        let positions: Vec<(f64, f64)> = (0..layout.n_qubits())
            .map(|q| {
                let z = layout.vertex(q);
                (z.re, z.im)
            })
            .collect();
        let mut edges: Vec<(usize, usize)> = layout
            .tree_edges()
            .iter()
            .chain(layout.non_tree_edges().iter())
            .map(|e| (e.a.min(e.b), e.a.max(e.b)))
            .collect();
        edges.sort_unstable();
        edges.dedup();
        Self { positions, edges }
    }
}

/// Magnetic gauge: which representative vector potential `A` (with
/// `curl A = B`, the same uniform field) is integrated along each edge
/// to produce the Peierls phase.
///
/// On a simply-connected (open-boundary) patch all gauges that share
/// the same per-plaquette flux give the **same spectrum** — they are
/// related by a diagonal unitary `U_i = exp(i χ_i)`. The
/// `gauge_invariance` test exercises exactly this, and M2's
/// closed-surface construction relies on it (§2.2: recomputing φ in a
/// different gauge must leave the spectrum invariant to ≤ 1e-9).
///
/// Both variants here use the **Euclidean embedding chart** (flat area
/// form `dx∧dy`). The hyperbolic area-form gauge — flux ∝ hyperbolic
/// area `4 dx∧dy/(1−r²)²`, integrated along geodesic edges — is the
/// M2 refinement and is a separate variant added there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gauge {
    /// Landau gauge, `A = (−B y, 0)`. Edge phase
    /// `φ_uv = −2π Φ · y_avg · Δx`.
    EmbeddingLandau,
    /// Symmetric gauge, `A = (B/2)(−y, x)`. Edge phase
    /// `φ_uv = π Φ · (x_u y_v − x_v y_u)` (the line integral of
    /// `(B/2)(−y dx + x dy)` along the straight `u → v` segment, with
    /// `B = 2π Φ` per unit embedding-chart area).
    EmbeddingSymmetric,
}

/// Peierls phase for the directed edge `u → v` under `gauge` at flux
/// ratio `flux` (`Φ/Φ₀`). Positions are embedding-chart coordinates.
#[must_use]
fn edge_phase(gauge: Gauge, pu: (f64, f64), pv: (f64, f64), flux: f64) -> f64 {
    let (xu, yu) = pu;
    let (xv, yv) = pv;
    match gauge {
        Gauge::EmbeddingLandau => {
            let dx = xv - xu;
            let y_avg = 0.5 * (yu + yv);
            -2.0 * std::f64::consts::PI * flux * y_avg * dx
        }
        Gauge::EmbeddingSymmetric => {
            std::f64::consts::PI * flux * (xu * yv - xv * yu)
        }
    }
}

/// Build the tight-binding magnetic Hamiltonian `H_ij = -t · exp(i φ_ij)`
/// for every edge `(i, j)`, with Peierls phases from `gauge` at flux
/// ratio `flux` (in units of `Φ/Φ₀`).
///
/// Returns an `n × n` Hermitian `Mat<c64>` ready for
/// `self_adjoint_eigen`.
#[must_use]
pub fn magnetic_hamiltonian(graph: &EmbeddedGraph, t: f64, flux: f64, gauge: Gauge) -> Mat<c64> {
    let n = graph.n_vertices();
    let mut h: Mat<c64> = Mat::zeros(n, n);
    for &(u, v) in &graph.edges {
        // Orientation u → v (with u < v by construction).
        let phi = edge_phase(gauge, graph.positions[u], graph.positions[v], flux);
        let amp = Complex64::from_polar(-t, phi); // -t · exp(i φ)
        h[(u, v)] = c64::new(amp.re, amp.im);
        h[(v, u)] = c64::new(amp.re, -amp.im);
    }
    h
}

/// Back-compatible wrapper: Landau gauge on the embedding chart.
/// Equivalent to `magnetic_hamiltonian(graph, t, flux, Gauge::EmbeddingLandau)`.
#[must_use]
pub fn magnetic_hamiltonian_landau(graph: &EmbeddedGraph, t: f64, flux: f64) -> Mat<c64> {
    magnetic_hamiltonian(graph, t, flux, Gauge::EmbeddingLandau)
}

/// Compute the eigenvalues of a Hermitian matrix. Wraps faer's
/// `self_adjoint_eigen` and returns the real parts (the imaginary
/// parts are zero up to FP precision for a true Hermitian input).
#[must_use]
pub fn hermitian_eigenvalues(h: &Mat<c64>) -> Vec<f64> {
    let n = h.nrows();
    let eig = h
        .self_adjoint_eigen(Side::Lower)
        .expect("self_adjoint_eigen on Hermitian matrix should succeed");
    let s = eig.S();
    let mut eigs: Vec<f64> = (0..n).map(|i| s[i].re).collect();
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    eigs
}

/// One row of the Hofstadter butterfly: flux ratio and the full
/// ordered eigenvalue spectrum at that flux.
#[derive(Clone, Debug)]
pub struct ButterflyRow {
    pub flux: f64,
    pub eigenvalues: Vec<f64>,
}

/// Compute the Hofstadter butterfly: sweep flux from `flux_min` to
/// `flux_max` in `n_steps + 1` linearly-spaced points (inclusive),
/// build the magnetic Hamiltonian via Landau gauge at each, and
/// diagonalise. Returns one `ButterflyRow` per flux value.
///
/// For canonical Hofstadter visualisation: `flux_min = 0`, `flux_max = 1`,
/// `n_steps` in [200, 1000] gives clean fractal resolution.
#[must_use]
pub fn hofstadter_butterfly(
    graph: &EmbeddedGraph,
    t: f64,
    flux_min: f64,
    flux_max: f64,
    n_steps: usize,
) -> Vec<ButterflyRow> {
    assert!(n_steps >= 1, "n_steps must be ≥ 1");
    let mut rows = Vec::with_capacity(n_steps + 1);
    let step = (flux_max - flux_min) / n_steps as f64;
    for k in 0..=n_steps {
        let flux = flux_min + k as f64 * step;
        let h = magnetic_hamiltonian_landau(graph, t, flux);
        let eigs = hermitian_eigenvalues(&h);
        rows.push(ButterflyRow {
            flux,
            eigenvalues: eigs,
        });
    }
    rows
}

/// Serialise a butterfly as CSV with columns `flux,eigenvalue`. One
/// row per (flux, eigenvalue) pair, suitable for matplotlib `scatter`
/// or `plot`. Returns the CSV as a `String`; the caller decides where
/// to write it.
#[must_use]
pub fn butterfly_to_csv(rows: &[ButterflyRow]) -> String {
    let mut s = String::from("flux,eigenvalue\n");
    for row in rows {
        for &e in &row.eigenvalues {
            s.push_str(&format!("{:.10},{:.10}\n", row.flux, e));
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4×4 square lattice has 16 vertices, 12 horizontal + 12 vertical
    /// = 24 edges. Spot-checks the `square_lattice` constructor.
    #[test]
    fn square_lattice_4x4_has_16_vertices_24_edges() {
        let g = EmbeddedGraph::square_lattice(4, 4);
        assert_eq!(g.n_vertices(), 16);
        assert_eq!(g.edges.len(), 24);
        // Corner (0,0) at index 0, opposite corner (3,3) at index 15.
        assert_eq!(g.positions[0], (0.0, 0.0));
        assert_eq!(g.positions[15], (3.0, 3.0));
    }

    /// At zero flux the magnetic Hamiltonian reduces to the standard
    /// (real) tight-binding Hamiltonian `H = -t A`, where `A` is the
    /// graph adjacency matrix. All matrix entries are pure real.
    #[test]
    fn zero_flux_hamiltonian_is_real() {
        let g = EmbeddedGraph::square_lattice(4, 4);
        let h = magnetic_hamiltonian_landau(&g, 1.0, 0.0);
        for i in 0..h.nrows() {
            for j in 0..h.ncols() {
                assert!(
                    h[(i, j)].im.abs() < 1e-15,
                    "(i,j)=({i},{j}) Im={} not zero at flux=0",
                    h[(i, j)].im
                );
            }
        }
    }

    /// **Bipartite-spectrum symmetry around `E = 0`.** A bipartite
    /// graph (which the square lattice is) at *any* flux has spectrum
    /// `{E_k}` invariant under `E → −E`. This is a strong
    /// Peierls-substitution sanity check: the gauge transformation
    /// `c_i → (-1)^{x_i + y_i} c_i` on the A sublattice flips `H → −H`,
    /// so eigenvalues come in `(+E, −E)` pairs for *every* flux.
    /// Pinned at flux = 1/3 (a non-trivial, non-rational-1/2 value).
    #[test]
    fn square_lattice_bipartite_spectrum_symmetry_at_flux_one_third() {
        let g = EmbeddedGraph::square_lattice(6, 6);
        let h = magnetic_hamiltonian_landau(&g, 1.0, 1.0 / 3.0);
        let eigs = hermitian_eigenvalues(&h);
        // Symmetry: for each eigenvalue E_k, -E_k must also be in the
        // spectrum (to within FP precision).
        let n = eigs.len();
        for k in 0..n {
            let target = -eigs[k];
            let found = eigs
                .iter()
                .any(|&e| (e - target).abs() < 1e-9);
            assert!(
                found,
                "eigenvalue {:.6} has no partner at {:.6} in spectrum",
                eigs[k], target
            );
        }
    }

    /// Eigenvalue count equals vertex count for every flux. Pinned
    /// at several rational flux values that exercise different
    /// regimes (0, 1/2, irrational).
    #[test]
    fn eigenvalue_count_equals_n_at_several_fluxes() {
        let g = EmbeddedGraph::square_lattice(5, 5);
        for &flux in &[0.0_f64, 0.25, 0.5, 1.0 / 7.0, 0.5_f64.sqrt() - 0.5] {
            let h = magnetic_hamiltonian_landau(&g, 1.0, flux);
            let eigs = hermitian_eigenvalues(&h);
            assert_eq!(eigs.len(), g.n_vertices(), "flux={flux}");
        }
    }

    /// **Gauge invariance of the spectrum (M0 deliverable).** On a
    /// simply-connected open-boundary patch, Landau and symmetric
    /// gauge encode the *same* uniform flux and must give the *same*
    /// spectrum (they differ by a diagonal unitary). This is the
    /// invariance checker M2's closed-surface Peierls construction
    /// relies on (§2.2: a different gauge must leave the spectrum
    /// invariant to ≤ 1e-9). Verified across several flux values; the
    /// per-edge phases genuinely differ between the two gauges (also
    /// asserted) so the test is not a tautology.
    #[test]
    fn spectrum_is_gauge_invariant_landau_vs_symmetric() {
        let g = EmbeddedGraph::square_lattice(6, 6);
        for &flux in &[0.0_f64, 0.1, 1.0 / 3.0, 0.5, 0.777] {
            let h_l = magnetic_hamiltonian(&g, 1.0, flux, Gauge::EmbeddingLandau);
            let h_s = magnetic_hamiltonian(&g, 1.0, flux, Gauge::EmbeddingSymmetric);
            let e_l = hermitian_eigenvalues(&h_l);
            let e_s = hermitian_eigenvalues(&h_s);
            for (k, (&a, &b)) in e_l.iter().zip(e_s.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-9,
                    "flux={flux} k={k}: Landau {a:.10} vs symmetric {b:.10}",
                );
            }
            // Confirm the gauges really differ at the edge level
            // (otherwise the spectral match would be trivial). Skip
            // flux=0 where both phases vanish.
            if flux != 0.0 {
                let differ = g.edges.iter().any(|&(u, v)| {
                    let pl = edge_phase(Gauge::EmbeddingLandau, g.positions[u], g.positions[v], flux);
                    let ps =
                        edge_phase(Gauge::EmbeddingSymmetric, g.positions[u], g.positions[v], flux);
                    (pl - ps).abs() > 1e-6
                });
                assert!(differ, "flux={flux}: gauges identical at edge level — test is vacuous");
            }
        }
    }

    /// **Zero-flux tight-binding spectrum** on square lattice with
    /// open boundary: eigenvalues are `E_{n,m} = -2t [cos(π n / (R+1)) + cos(π m / (C+1))]`
    /// for `n ∈ 1..=R`, `m ∈ 1..=C`. Spot-check the lowest and
    /// highest eigenvalues for a 4×4 lattice against this analytic
    /// formula.
    #[test]
    fn zero_flux_square_lattice_matches_analytic_tight_binding() {
        let (r, c) = (4, 4);
        let g = EmbeddedGraph::square_lattice(r, c);
        let h = magnetic_hamiltonian_landau(&g, 1.0, 0.0);
        let eigs = hermitian_eigenvalues(&h);

        // Analytic spectrum (OBC, hopping t = 1, sign convention H = -t A).
        let mut analytic: Vec<f64> = Vec::with_capacity(r * c);
        for n in 1..=r {
            for m in 1..=c {
                let theta_n = std::f64::consts::PI * n as f64 / (r as f64 + 1.0);
                let theta_m = std::f64::consts::PI * m as f64 / (c as f64 + 1.0);
                analytic.push(-2.0 * (theta_n.cos() + theta_m.cos()));
            }
        }
        analytic.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for (k, (&num, &ana)) in eigs.iter().zip(analytic.iter()).enumerate() {
            assert!(
                (num - ana).abs() < 1e-10,
                "eig {k}: numeric {num:.10} vs analytic {ana:.10}",
            );
        }
    }

    /// **Hofstadter periodicity:** the spectrum at flux Φ equals the
    /// spectrum at flux `Φ + 1` (the magnetic translation group sees
    /// flux only mod 1). Pinned for several Φ on the square lattice.
    #[test]
    fn hofstadter_spectrum_is_periodic_in_flux_with_period_1() {
        let g = EmbeddedGraph::square_lattice(4, 4);
        for &flux in &[0.0_f64, 0.13, 0.31, 0.5, 0.7] {
            let eigs_0 = hermitian_eigenvalues(&magnetic_hamiltonian_landau(&g, 1.0, flux));
            let eigs_1 = hermitian_eigenvalues(&magnetic_hamiltonian_landau(&g, 1.0, flux + 1.0));
            for (k, (&a, &b)) in eigs_0.iter().zip(eigs_1.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-9,
                    "flux={flux}, k={k}: E(Φ)={a:.6} ≠ E(Φ+1)={b:.6}",
                );
            }
        }
    }

    /// **Hofstadter butterfly sweep** produces the expected number
    /// of rows and per-row eigenvalue count. Smoke-test of the
    /// public `hofstadter_butterfly` driver.
    #[test]
    fn hofstadter_butterfly_smoke_test() {
        let g = EmbeddedGraph::square_lattice(4, 4);
        let rows = hofstadter_butterfly(&g, 1.0, 0.0, 1.0, 50);
        assert_eq!(rows.len(), 51); // inclusive at both endpoints
        for row in &rows {
            assert_eq!(row.eigenvalues.len(), 16);
        }
        // Bipartite symmetry holds at every step.
        for row in &rows {
            for &e in &row.eigenvalues {
                assert!(
                    row.eigenvalues.iter().any(|&f| (f + e).abs() < 1e-9),
                    "flux {}: eigenvalue {e:.6} has no partner",
                    row.flux
                );
            }
        }
    }

    /// **EmbeddedGraph::from(&HyperbolicLayout)** correctness:
    /// vertex count matches, position is the Poincaré-disk
    /// coordinate, edge list is the union of tree + non-tree edges
    /// from the layout.
    #[test]
    fn embedded_graph_from_hyperbolic_layout_preserves_structure() {
        let layout = HyperbolicLayout::pq_tiling(7, 3, 1);
        let g = EmbeddedGraph::from(&layout);
        assert_eq!(g.n_vertices(), layout.n_qubits());
        assert_eq!(g.n_vertices(), 35); // hand-verified earlier
        // Position equals Poincaré coordinate of the vertex.
        for q in 0..layout.n_qubits() {
            let z = layout.vertex(q);
            assert_eq!(g.positions[q], (z.re, z.im));
        }
        // Edge count: tree (n-1) + non-tree, no duplicates.
        let expected = layout.tree_edges().len() + layout.non_tree_edges().len();
        assert_eq!(g.edges.len(), expected);
    }

    /// **First hyperbolic Hofstadter sanity check:** {7, 3} at
    /// radius 1 (35 vertices) at zero flux must reproduce the
    /// standard graph-Laplacian-like tight-binding spectrum;
    /// eigenvalues should be symmetric around 0 only if the
    /// graph is bipartite, which the vertex graph of {7, 3} is
    /// *not* (it contains odd cycles — the heptagonal faces).
    /// So this test pins the *non-symmetric* zero-flux spectrum
    /// to confirm the bipartite-symmetry test above is checking
    /// a real property and not just a tautology.
    #[test]
    fn hyperbolic_7_3_zero_flux_is_not_bipartite_symmetric() {
        let layout = HyperbolicLayout::pq_tiling(7, 3, 1);
        let g = EmbeddedGraph::from(&layout);
        let h = magnetic_hamiltonian_landau(&g, 1.0, 0.0);
        let eigs = hermitian_eigenvalues(&h);
        // Look for at least one eigenvalue without an opposite-sign
        // partner — that proves {7,3} is non-bipartite (odd 7-cycles
        // forbid bipartiteness).
        let any_unpaired = eigs
            .iter()
            .any(|&e| eigs.iter().all(|&f| (f + e).abs() > 1e-6));
        assert!(
            any_unpaired,
            "{{7,3}} vertex graph appears bipartite-symmetric at zero flux \
             — this contradicts its non-bipartiteness (odd heptagonal cycles)",
        );
    }
}
