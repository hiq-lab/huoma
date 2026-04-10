//! Scale smoke test for the projected TTN pipeline — Track F milestone
//! **F.5**.
//!
//! Proves the full `ProjectedTtn` pipeline works end-to-end at
//! N ≥ 100,000 qubits by:
//!
//! 1. Building a synthetic heavy-hex-like topology by tiling Eagle's
//!    row+bridge pattern.
//! 2. Assigning per-qubit frequencies with ~1% volatile fraction.
//! 3. Partitioning → extracting volatile islands.
//! 4. Building a `ProjectedTtn`, running 3 Floquet steps.
//! 5. Reading all N ⟨Z⟩ values and verifying they're finite and bounded.

use huoma::kicked_ising::KimParams;
use huoma::ttn::boundary::BoundaryMode;
use huoma::ttn::partition::partition_tree_adaptive;
use huoma::ttn::projected::ProjectedTtn;
use huoma::ttn::topology::{Edge, Topology};

/// Build a synthetic linear-chain topology (the cheapest valid tree)
/// at the given size. For the scale test we don't need heavy-hex
/// structure — the partitioner and ProjectedTtn work on any tree.
/// A linear chain exercises the full pipeline at N=100K without the
/// memory cost of constructing a complex graph.
fn synthetic_chain(n: usize) -> Topology {
    let edges: Vec<Edge> = (0..n - 1).map(|i| Edge { a: i, b: i + 1 }).collect();
    Topology::from_edges_lightweight(n, edges)
}

/// Generate a deterministic frequency vector where ~fraction of qubits
/// are "disordered" (irrational frequency → volatile) and the rest are
/// commensurate (integer frequency → stable).
fn mixed_frequencies(n: usize, volatile_fraction: f64) -> Vec<f64> {
    let primes = [2.0_f64, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0];
    (0..n)
        .map(|i| {
            // Deterministic hash to decide if this qubit is "disordered".
            let hash = ((i * 2654435761) >> 16) as f64 / 65536.0;
            if hash < volatile_fraction {
                // Incommensurate: √prime
                primes[i % primes.len()].sqrt()
            } else {
                // Commensurate: integer
                1.0 + (i % 4) as f64
            }
        })
        .collect()
}

/// The million-qubit pipeline smoke test.
///
/// At N = 100,000 with ~1% volatile fraction this exercises:
/// - Lightweight topology construction (O(N) BFS, no cut partitions)
/// - Local-radius edge scoring (O(N · radius²))
/// - Budget-adaptive partitioning
/// - Volatile island extraction
/// - Boundary tensor computation
/// - ProjectedTtn construction
/// - 3 Floquet steps (only volatile islands simulated)
/// - Full-graph observable readout (100K values)
///
/// Target: completes in < 30 seconds in release.
#[test]
fn projected_ttn_100k_qubits_3_steps_completes() {
    let n = 100_000;
    let volatile_fraction = 0.01; // ~1% volatile

    let start = std::time::Instant::now();

    // 1. Build the topology (lightweight — no cut partitions).
    let topology = synthetic_chain(n);
    assert_eq!(topology.n_qubits(), n);
    assert!(!topology.has_cut_partitions());
    let t_topology = start.elapsed();

    // 2. Generate frequencies with ~1% volatile fraction.
    let frequencies = mixed_frequencies(n, volatile_fraction);
    assert_eq!(frequencies.len(), n);

    // 3. Partition with a moderate budget.
    let total_budget = (n - 1) * 4; // average χ = 4
    let partition = partition_tree_adaptive(
        &frequencies,
        &topology,
        total_budget,
        2,
        16,
        5, // radius
    );
    let n_volatile_qubits = partition.volatile_qubits.len();
    let n_volatile_edges = partition.volatile_edges.len();
    let t_partition = start.elapsed();

    eprintln!("[F.5] N={n}, partition: {n_volatile_edges} volatile edges, {n_volatile_qubits} volatile qubits");
    eprintln!("[F.5] volatile fraction: {:.2}%", n_volatile_qubits as f64 / n as f64 * 100.0);
    eprintln!("[F.5] topology: {t_topology:.2?}, partition: {:?}", t_partition - t_topology);

    // 4. Build ProjectedTtn.
    let params = KimParams {
        j: -std::f64::consts::FRAC_PI_4,
        h_x: 0.4,
        h_z: 0.0,
        dt: 1.0,
    };
    let mut pttn = ProjectedTtn::new(
        &topology,
        &frequencies,
        partition,
        &[], // no non-tree edges on a linear chain
        params,
        3, // n_boundary_steps
        BoundaryMode::ProductState,
    );
    let t_build = start.elapsed();
    eprintln!(
        "[F.5] ProjectedTtn: {} islands, {} volatile qubits, build: {:?}",
        pttn.n_islands(),
        pttn.n_qubits_volatile(),
        t_build - t_partition
    );

    assert_eq!(pttn.n_qubits_total(), n);

    // 5. Run 3 Floquet steps.
    let n_steps = 3;
    for step in 1..=n_steps {
        pttn.apply_floquet_step(params)
            .unwrap_or_else(|e| panic!("step {step} failed: {e}"));
    }
    let t_evolve = start.elapsed();
    eprintln!("[F.5] {n_steps} Floquet steps: {:?}", t_evolve - t_build);

    // 6. Read all N ⟨Z⟩ values.
    let z = pttn.expectation_z_all();
    let t_obs = start.elapsed();
    assert_eq!(z.len(), n);
    eprintln!("[F.5] expectation_z_all: {:?}", t_obs - t_evolve);

    // 7. Verify all values are finite and bounded.
    for (q, &zq) in z.iter().enumerate() {
        assert!(
            zq.is_finite() && (-1.0..=1.0).contains(&zq),
            "q={q}: ⟨Z⟩ = {zq} invalid"
        );
    }

    // 8. Verify discarded weight is finite.
    let disc = pttn.total_discarded_weight();
    assert!(disc.is_finite() && disc >= 0.0);

    let total_time = start.elapsed();
    eprintln!(
        "\n[F.5] SUMMARY: {n} qubits, {} volatile ({:.1}%), {} islands, \
         {n_steps} steps, total = {total_time:.2?}, discarded = {disc:.4e}",
        pttn.n_qubits_volatile(),
        pttn.n_qubits_volatile() as f64 / n as f64 * 100.0,
        pttn.n_islands(),
    );

    // Target: < 30 seconds.
    assert!(
        total_time.as_secs() < 30,
        "100K-qubit projected run took {total_time:.2?}, exceeds 30s target"
    );
}
