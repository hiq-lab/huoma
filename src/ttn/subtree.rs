//! Sub-topology extraction: given a [`TreePartition`], extract each
//! connected component of volatile edges as an independent
//! [`VolatileIsland`] with its own local [`Topology`] and qubit-index
//! remapping.
//!
//! Track F milestone **F.2**.
//!
//! Each island is a self-contained TTN simulation domain: it has its own
//! `Topology` (over local indices 0..n_island), a bidirectional index map
//! back to the full graph, and a list of boundary edges where the volatile
//! island meets the stable exterior. The [`crate::ttn::Ttn`] constructor
//! accepts the island's `Topology` directly, so building a TTN for each
//! island is a single `Ttn::new(island.topology.clone())` call.

use crate::ttn::partition::{EdgeClass, TreePartition};
use crate::ttn::topology::{Edge, EdgeId, Topology};

/// A boundary edge connecting a volatile island to the stable exterior.
#[derive(Debug, Clone)]
pub struct BoundaryEdge {
    /// The volatile-side qubit in **local** (island) indices.
    pub local_qubit: usize,
    /// The stable-side qubit in **global** (full-graph) indices.
    pub global_stable_qubit: usize,
    /// The edge's `EdgeId` in the **full** topology.
    pub full_edge_id: EdgeId,
}

/// One connected component of volatile edges, ready for TTN simulation.
#[derive(Debug, Clone)]
pub struct VolatileIsland {
    /// Sub-tree topology over local indices `0..n_island_qubits`.
    /// Constructed via [`Topology::from_edges`] (full build with cut
    /// partitions) because islands are small enough for the O(N²) cost.
    pub topology: Topology,

    /// Map from full-graph qubit index to island-local index. Length =
    /// full `n_qubits`. `None` for qubits not in this island.
    pub global_to_local: Vec<Option<usize>>,

    /// Map from island-local index to full-graph qubit index. Length =
    /// number of qubits in this island.
    pub local_to_global: Vec<usize>,

    /// Boundary edges connecting this island to the stable exterior.
    pub boundary_edges: Vec<BoundaryEdge>,

    /// Per-edge χ allocation for this island's edges, in local `EdgeId`
    /// order. Derived from the full partition's `chi_per_edge` via the
    /// edge index mapping.
    pub chi_per_edge: Vec<usize>,
}

/// Extract all connected components of volatile edges from a partitioned
/// topology, each as an independent [`VolatileIsland`].
///
/// # Algorithm
///
/// 1. Build an adjacency list over volatile qubits using only volatile edges.
/// 2. BFS to find connected components.
/// 3. For each component: collect the vertex set, renumber to `0..n`, build
///    edge list, construct `Topology::from_edges`, map boundary edges.
///
/// # Panics
///
/// Panics if the full topology's edge data is inconsistent with the
/// partition (e.g. an `EdgeId` in `partition.volatile_edges` is out of
/// range). This is a programmer error — the partition was produced from
/// the same topology.
#[must_use]
pub fn extract_volatile_islands(
    full_topology: &Topology,
    partition: &TreePartition,
) -> Vec<VolatileIsland> {
    let n_full = full_topology.n_qubits();

    if partition.volatile_edges.is_empty() {
        return Vec::new();
    }

    // Step 1: identify which qubits are volatile.
    let mut is_volatile = vec![false; n_full];
    for &q in &partition.volatile_qubits {
        is_volatile[q] = true;
    }

    // Step 2: build adjacency over volatile qubits using volatile edges.
    let mut vol_adj: Vec<Vec<(usize, EdgeId)>> = vec![Vec::new(); n_full];
    for &eid in &partition.volatile_edges {
        let e = full_topology.edge(eid);
        // Both endpoints of a volatile edge are volatile qubits.
        vol_adj[e.a].push((e.b, eid));
        vol_adj[e.b].push((e.a, eid));
    }

    // Step 3: BFS to find connected components.
    let mut visited = vec![false; n_full];
    let mut islands = Vec::new();

    for &seed in &partition.volatile_qubits {
        if visited[seed] {
            continue;
        }
        // BFS from seed.
        let mut component_qubits: Vec<usize> = Vec::new();
        let mut component_full_edges: Vec<EdgeId> = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        visited[seed] = true;
        queue.push_back(seed);
        while let Some(v) = queue.pop_front() {
            component_qubits.push(v);
            for &(w, eid) in &vol_adj[v] {
                if !visited[w] {
                    visited[w] = true;
                    queue.push_back(w);
                }
                // Record edge (dedup by only recording from the smaller endpoint).
                let e = full_topology.edge(eid);
                if v == e.a.min(e.b) {
                    component_full_edges.push(eid);
                }
            }
        }
        component_qubits.sort_unstable();
        component_full_edges.sort_unstable_by_key(|e| e.0);

        // Step 4: build local index map.
        let n_island = component_qubits.len();
        let mut global_to_local = vec![None; n_full];
        let mut local_to_global = Vec::with_capacity(n_island);
        for (local, &global) in component_qubits.iter().enumerate() {
            global_to_local[global] = Some(local);
            local_to_global.push(global);
        }

        // Step 5: build local edge list.
        let local_edges: Vec<Edge> = component_full_edges
            .iter()
            .map(|&eid| {
                let e = full_topology.edge(eid);
                let la = global_to_local[e.a].expect("volatile edge endpoint must be in island");
                let lb = global_to_local[e.b].expect("volatile edge endpoint must be in island");
                Edge { a: la, b: lb }
            })
            .collect();

        // Step 6: build local topology (full build — islands are small).
        let topology = Topology::from_edges(n_island, local_edges);

        // Step 7: map chi_per_edge from full partition to local order.
        // The local edges are in the same order as component_full_edges,
        // which is how Topology::from_edges was called, so local EdgeId(i)
        // corresponds to component_full_edges[i].
        let chi_per_edge: Vec<usize> = component_full_edges
            .iter()
            .map(|&eid| partition.chi_per_edge[eid.0])
            .collect();

        // Step 8: identify boundary edges. A boundary edge is a volatile
        // edge in this island where at least one endpoint also touches a
        // stable edge in the full graph. We scan the full topology's
        // neighbours of each island qubit for stable edges.
        let mut boundary_edges = Vec::new();
        for (local_q, &global_q) in local_to_global.iter().enumerate() {
            for &full_eid in full_topology.neighbours(global_q) {
                if partition.edge_classes[full_eid.0] == EdgeClass::Stable {
                    // This qubit touches a stable edge → it's a boundary qubit.
                    let full_e = full_topology.edge(full_eid);
                    let stable_side = if full_e.a == global_q {
                        full_e.b
                    } else {
                        full_e.a
                    };
                    boundary_edges.push(BoundaryEdge {
                        local_qubit: local_q,
                        global_stable_qubit: stable_side,
                        full_edge_id: full_eid,
                    });
                }
            }
        }

        islands.push(VolatileIsland {
            topology,
            global_to_local,
            local_to_global,
            boundary_edges,
            chi_per_edge,
        });
    }

    islands
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttn::partition::partition_tree_adaptive;
    use crate::ttn::heavy_hex::HeavyHexLayout;

    #[test]
    fn all_stable_produces_no_islands() {
        let topology = Topology::linear_chain(10);
        let frequencies = vec![1.0; 10]; // uniform → all stable
        let partition = partition_tree_adaptive(&frequencies, &topology, 72, 2, 16, 5);
        let islands = extract_volatile_islands(&topology, &partition);
        assert!(islands.is_empty());
    }

    #[test]
    fn all_volatile_produces_one_island() {
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
        let islands = extract_volatile_islands(&topology, &partition);
        assert_eq!(islands.len(), 1);
        assert_eq!(islands[0].topology.n_qubits(), 6);
        assert_eq!(islands[0].topology.n_edges(), 5);
        assert!(islands[0].boundary_edges.is_empty(), "no stable edges → no boundary");
    }

    #[test]
    fn index_map_round_trips() {
        let topology = Topology::linear_chain(8);
        let frequencies = [
            1.0,
            2.0_f64.sqrt(),
            3.0_f64.sqrt(),
            5.0_f64.sqrt(),
            7.0_f64.sqrt(),
            11.0_f64.sqrt(),
            1.0,
            2.0,
        ];
        let partition = partition_tree_adaptive(&frequencies, &topology, 50, 2, 16, 5);
        let islands = extract_volatile_islands(&topology, &partition);
        for island in &islands {
            for (local, &global) in island.local_to_global.iter().enumerate() {
                assert_eq!(
                    island.global_to_local[global],
                    Some(local),
                    "round-trip failed for global={global}, local={local}"
                );
            }
        }
    }

    #[test]
    fn island_topology_is_valid_tree() {
        let topology = Topology::linear_chain(10);
        let frequencies = [
            1.0, 2.0, 3.0,           // commensurate interior (stable)
            7.0_f64.sqrt(),           // transition
            11.0_f64.sqrt(),          // incommensurate
            13.0_f64.sqrt(),
            17.0_f64.sqrt(),
            1.0, 2.0, 3.0,           // commensurate tail (stable)
        ];
        let partition = partition_tree_adaptive(&frequencies, &topology, 40, 2, 16, 5);
        let islands = extract_volatile_islands(&topology, &partition);
        for island in &islands {
            // Topology::from_edges validates tree invariants at construction.
            // If we got here, the island topology is a valid tree.
            assert_eq!(island.topology.n_edges(), island.topology.n_qubits() - 1);
            assert!(island.topology.n_qubits() > 0);
        }
    }

    #[test]
    fn eagle_127q_disordered_produces_islands_with_boundaries() {
        let layout = HeavyHexLayout::ibm_eagle_127();
        let primes = [2.0_f64, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0];
        let frequencies: Vec<f64> = (0..127)
            .map(|i| 0.8 + 0.05 * primes[i % primes.len()].sqrt())
            .collect();
        let partition = partition_tree_adaptive(&frequencies, layout.tree(), 504, 2, 32, 5);
        let islands = extract_volatile_islands(layout.tree(), &partition);

        // Should have at least one island.
        assert!(!islands.is_empty(), "expected volatile islands on disordered Eagle");

        // Total volatile qubits across all islands should match the partition.
        let total_volatile: usize = islands.iter().map(|i| i.topology.n_qubits()).sum();
        assert_eq!(total_volatile, partition.volatile_qubits.len());

        // At least one island should have boundary edges.
        let total_boundary: usize = islands.iter().map(|i| i.boundary_edges.len()).sum();
        assert!(total_boundary > 0, "expected boundary edges on disordered Eagle");

        // Chi allocation matches.
        for island in &islands {
            assert_eq!(island.chi_per_edge.len(), island.topology.n_edges());
            for &chi in &island.chi_per_edge {
                assert!(chi >= 2 && chi <= 32, "chi out of bounds: {chi}");
            }
        }
    }

    #[test]
    fn island_chi_allocation_sums_to_full_volatile_budget() {
        let topology = Topology::linear_chain(8);
        let frequencies = [
            1.0,
            2.0_f64.sqrt(),
            3.0_f64.sqrt(),
            5.0_f64.sqrt(),
            7.0_f64.sqrt(),
            11.0_f64.sqrt(),
            1.0,
            2.0,
        ];
        let total_budget = 50;
        let partition = partition_tree_adaptive(&frequencies, &topology, total_budget, 2, 16, 5);
        let islands = extract_volatile_islands(&topology, &partition);
        let island_budget: usize = islands
            .iter()
            .flat_map(|i| i.chi_per_edge.iter())
            .sum();
        let partition_volatile_budget: usize = partition
            .chi_per_edge
            .iter()
            .filter(|&&c| c > 0)
            .sum();
        assert_eq!(
            island_budget, partition_volatile_budget,
            "island chi sum must match partition volatile budget"
        );
    }
}
