//! Two-site merge, bipartition SVD, and whole-tree contraction.
//!
//! The main routine is [`apply_two_qubit_on_edge_native`]: it contracts two
//! adjacent site tensors along their shared edge into a single Θ tensor,
//! applies a 4×4 gate to the pair of physical legs, and splits Θ back into
//! two sites via a bipartition SVD with truncation. The truncation error is
//! counted into the caller's discarded-weight accumulator and the new bond
//! dimension is returned.
//!
//! [`tree_to_statevector`] walks the tree BFS-style from vertex 0 and
//! contracts each new site into a growing accumulator, producing the full
//! `2^N` statevector with qubits ordered `(q0, q1, …, q_{n-1})`. It is used
//! by the native `Ttn::expectation_z` implementation and by the D.2 Y-junction
//! / star tests that compare against `DenseState`.
//!
//! Track D milestone D.2.

use faer::Mat;
use num_complex::Complex64;

use crate::error::Result;
use crate::mps::TruncationMode;

use super::site::{flatten_tensor_raw, TtnSite};
use super::topology::{EdgeId, Topology};

type C = Complex64;

/// Apply a 4×4 two-qubit gate `u_gate` on the edge connecting `sites[u]` and
/// `sites[v]`, where `(u, v)` are the endpoints of `edge_id`. Truncates the
/// new bond at `max_bond` (or by `mode`) and returns the new bond dimension
/// and the discarded-weight contribution (sum of squared dropped singular
/// values).
///
/// The routine does not require the orthogonality center to sit at either
/// endpoint; the caller is expected to move the center appropriately for
/// the truncation to be optimal in the 2-norm sense. See the Ttn wrapper in
/// `mod.rs` for the gauge choreography.
pub(super) fn apply_two_qubit_on_edge_native(
    sites: &mut [TtnSite],
    topology: &Topology,
    edge_id: EdgeId,
    u_gate: [[C; 4]; 4],
    max_bond: usize,
    mode: TruncationMode,
) -> Result<(usize, f64)> {
    let edge = topology.edge(edge_id);
    let u = edge.a;
    let v = edge.b;

    // --- Flatten site[u] as [rest_u_nonphys…, physical_u, e] ---
    let ax_u_e = sites[u].axis_for_edge(edge_id);
    let phys_u = sites[u].physical_axis();
    let u_rank = sites[u].rank();
    let rest_u_nonphys_axes: Vec<usize> = (0..u_rank)
        .filter(|&ax| ax != ax_u_e && ax != phys_u)
        .collect();
    let mut row_axes_u: Vec<usize> = rest_u_nonphys_axes.clone();
    row_axes_u.push(phys_u);
    let (u_flat, rows_u, cols_u) = sites[u].flatten_to_matrix(&row_axes_u, &[ax_u_e]);
    let rest_u_nonphys_flat = rows_u / 2;
    let dim_e = cols_u;

    // --- Flatten site[v] as [e, rest_v_nonphys…, physical_v] ---
    let ax_v_e = sites[v].axis_for_edge(edge_id);
    let phys_v = sites[v].physical_axis();
    let v_rank = sites[v].rank();
    let rest_v_nonphys_axes: Vec<usize> = (0..v_rank)
        .filter(|&ax| ax != ax_v_e && ax != phys_v)
        .collect();
    let mut col_axes_v: Vec<usize> = rest_v_nonphys_axes.clone();
    col_axes_v.push(phys_v);
    let (v_flat, rows_v_inner, cols_v) = sites[v].flatten_to_matrix(&[ax_v_e], &col_axes_v);
    debug_assert_eq!(rows_v_inner, dim_e, "shared edge dims must match");
    let rest_v_nonphys_flat = cols_v / 2;

    // --- Contract U_mat · V_mat = Θ_mat [rows_u × cols_v] ---
    let u_mat = Mat::from_fn(rows_u, dim_e, |i, j| {
        let c = u_flat[i * dim_e + j];
        faer::c64::new(c.re, c.im)
    });
    let v_mat = Mat::from_fn(dim_e, cols_v, |i, j| {
        let c = v_flat[i * cols_v + j];
        faer::c64::new(c.re, c.im)
    });
    let theta_mat = &u_mat * &v_mat;

    // Theta[row_idx * cols_v + col_idx] with
    //   row_idx = rest_u_nonphys_idx * 2 + sigma_u
    //   col_idx = rest_v_nonphys_idx * 2 + sigma_v
    let mut theta_flat = vec![C::new(0.0, 0.0); rows_u * cols_v];
    for i in 0..rows_u {
        for j in 0..cols_v {
            let z = theta_mat[(i, j)];
            theta_flat[i * cols_v + j] = C::new(z.re, z.im);
        }
    }

    // --- Apply the gate in place on the (sigma_u, sigma_v) sub-block of
    // every (rest_u_nonphys_idx, rest_v_nonphys_idx) pair.
    // Gate convention: u_gate[sp_u*2 + sp_v][s_u*2 + s_v].
    let mut theta_new = vec![C::new(0.0, 0.0); theta_flat.len()];
    for ru in 0..rest_u_nonphys_flat {
        for rv in 0..rest_v_nonphys_flat {
            let th00 = theta_flat[(ru * 2) * cols_v + (rv * 2)];
            let th01 = theta_flat[(ru * 2) * cols_v + (rv * 2 + 1)];
            let th10 = theta_flat[(ru * 2 + 1) * cols_v + (rv * 2)];
            let th11 = theta_flat[(ru * 2 + 1) * cols_v + (rv * 2 + 1)];
            for sp_u in 0..2 {
                for sp_v in 0..2 {
                    let row_idx = sp_u * 2 + sp_v;
                    let new_val = u_gate[row_idx][0] * th00
                        + u_gate[row_idx][1] * th01
                        + u_gate[row_idx][2] * th10
                        + u_gate[row_idx][3] * th11;
                    theta_new[(ru * 2 + sp_u) * cols_v + (rv * 2 + sp_v)] = new_val;
                }
            }
        }
    }

    // --- SVD of Θ_new with row = (rest_u_nonphys, sigma_u) and col = (rest_v_nonphys, sigma_v). ---
    let theta_faer = Mat::from_fn(rows_u, cols_v, |i, j| {
        let c = theta_new[i * cols_v + j];
        faer::c64::new(c.re, c.im)
    });
    let svd = theta_faer
        .thin_svd()
        .map_err(|_| crate::error::ProjError::SvdFailed(0))?;
    let u_svd = svd.U();
    let s = svd.S();
    let v_svd = svd.V();
    let n_sv = s.column_vector().nrows();
    let cap = max_bond.min(n_sv).min(rows_u).min(cols_v);

    // Decide how many singular values to keep and compute discarded weight.
    let mut keep = cap;
    match mode {
        TruncationMode::Absolute => {
            if cap > 1 {
                let s_max = s.column_vector()[0].re;
                if s_max > 1e-15 {
                    for i in (1..cap).rev() {
                        if s.column_vector()[i].re / s_max < 1e-14 {
                            keep = i;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        TruncationMode::DiscardedWeight { eps } => {
            let total: f64 = (0..n_sv)
                .map(|i| {
                    let v = s.column_vector()[i].re;
                    v * v
                })
                .sum();
            if total > 1e-30 {
                let target = (1.0 - eps) * total;
                let mut cum = 0.0_f64;
                let mut k = 0usize;
                for i in 0..cap {
                    let v = s.column_vector()[i].re;
                    cum += v * v;
                    k = i + 1;
                    if cum >= target {
                        break;
                    }
                }
                keep = k;
            }
        }
    }
    let actual_chi = keep.max(1);

    // Discarded weight = Σ σ_i² over dropped SVs (including those past `cap`
    // if the hard budget clipped us).
    let mut discarded = 0.0_f64;
    for i in actual_chi..n_sv {
        let v = s.column_vector()[i].re;
        discarded += v * v;
    }

    // --- Build new U matrix with S absorbed: new_u[i, k] = U_svd[i, k] · s[k]. ---
    // This makes site[u] the bearer of the full state; site[v] is left as an
    // isometry in V^H shape. Callers that want the center on v should
    // swap the absorption side; for now we hard-code "absorb into u".
    let mut new_u_mat = vec![C::new(0.0, 0.0); rows_u * actual_chi];
    for i in 0..rows_u {
        for k in 0..actual_chi {
            let uval = u_svd[(i, k)];
            let sval = s.column_vector()[k].re;
            new_u_mat[i * actual_chi + k] = C::new(uval.re * sval, uval.im * sval);
        }
    }

    // --- Build new V matrix: new_v[k, j] = V_svd^H[k, j] = conj(V_svd[j, k]). ---
    let mut new_v_mat = vec![C::new(0.0, 0.0); actual_chi * cols_v];
    for k in 0..actual_chi {
        for j in 0..cols_v {
            let vval = v_svd[(j, k)];
            new_v_mat[k * cols_v + j] = C::new(vval.re, -vval.im);
        }
    }

    // --- Rebuild site[u]. ---
    // Intermediate layout: [rest_u_nonphys_axes…, phys, new_e].
    // Need to permute back to the original site[u] axis ordering, with the
    // e-axis's dimension updated to actual_chi.
    let n_rest_u = rest_u_nonphys_axes.len();
    let mut intermediate_shape_u: Vec<usize> = rest_u_nonphys_axes
        .iter()
        .map(|&ax| sites[u].shape[ax])
        .collect();
    intermediate_shape_u.push(2);
    intermediate_shape_u.push(actual_chi);
    // intermediate_edges_u has length (n_rest_u + 2) - 1 = n_rest_u + 1
    // since physical is at position n_rest_u (not in edges).
    let mut intermediate_edges_u: Vec<EdgeId> = rest_u_nonphys_axes
        .iter()
        .map(|&ax| sites[u].edges[ax])
        .collect();
    intermediate_edges_u.push(edge_id);
    let intermediate_u = TtnSite::unflatten_from_matrix(
        new_u_mat,
        intermediate_shape_u,
        intermediate_edges_u,
    );
    // Target axis order (in site[u]'s original numbering) → intermediate axis.
    let target_order_u: Vec<usize> = (0..u_rank)
        .map(|o| {
            if o == phys_u {
                n_rest_u
            } else if o == ax_u_e {
                n_rest_u + 1
            } else {
                rest_u_nonphys_axes
                    .iter()
                    .position(|&x| x == o)
                    .expect("o must be in rest_u_nonphys")
            }
        })
        .collect();
    let needs_permute_u = target_order_u.iter().enumerate().any(|(i, &j)| i != j);
    let site_u_new = if needs_permute_u {
        let (permuted, tot, one) = intermediate_u.flatten_to_matrix(&target_order_u, &[]);
        debug_assert_eq!(one, 1);
        debug_assert_eq!(tot, intermediate_u.len());
        let final_shape: Vec<usize> = target_order_u
            .iter()
            .map(|&i| intermediate_u.shape[i])
            .collect();
        TtnSite::unflatten_from_matrix(permuted, final_shape, sites[u].edges.clone())
    } else {
        TtnSite::unflatten_from_matrix(
            intermediate_u.data,
            intermediate_u.shape,
            sites[u].edges.clone(),
        )
    };
    sites[u] = site_u_new;

    // --- Rebuild site[v]. ---
    // Intermediate layout: [new_e, rest_v_nonphys_axes…, phys].
    let n_rest_v = rest_v_nonphys_axes.len();
    let mut intermediate_shape_v: Vec<usize> = Vec::with_capacity(n_rest_v + 2);
    intermediate_shape_v.push(actual_chi);
    for &ax in &rest_v_nonphys_axes {
        intermediate_shape_v.push(sites[v].shape[ax]);
    }
    intermediate_shape_v.push(2);
    let mut intermediate_edges_v: Vec<EdgeId> = Vec::with_capacity(n_rest_v + 1);
    intermediate_edges_v.push(edge_id);
    for &ax in &rest_v_nonphys_axes {
        intermediate_edges_v.push(sites[v].edges[ax]);
    }
    let intermediate_v = TtnSite::unflatten_from_matrix(
        new_v_mat,
        intermediate_shape_v,
        intermediate_edges_v,
    );
    let target_order_v: Vec<usize> = (0..v_rank)
        .map(|o| {
            if o == phys_v {
                n_rest_v + 1
            } else if o == ax_v_e {
                0
            } else {
                1 + rest_v_nonphys_axes
                    .iter()
                    .position(|&x| x == o)
                    .expect("o must be in rest_v_nonphys")
            }
        })
        .collect();
    let needs_permute_v = target_order_v.iter().enumerate().any(|(i, &j)| i != j);
    let site_v_new = if needs_permute_v {
        let (permuted, tot, one) = intermediate_v.flatten_to_matrix(&target_order_v, &[]);
        debug_assert_eq!(one, 1);
        debug_assert_eq!(tot, intermediate_v.len());
        let final_shape: Vec<usize> = target_order_v
            .iter()
            .map(|&i| intermediate_v.shape[i])
            .collect();
        TtnSite::unflatten_from_matrix(permuted, final_shape, sites[v].edges.clone())
    } else {
        TtnSite::unflatten_from_matrix(
            intermediate_v.data,
            intermediate_v.shape,
            sites[v].edges.clone(),
        )
    };
    sites[v] = site_v_new;

    Ok((actual_chi, discarded))
}

/// Axis label for the `tree_to_statevector` accumulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AxLabel {
    /// Virtual edge that hasn't been contracted yet.
    Edge(usize),
    /// Physical leg of the given qubit (already contracted into the accumulator).
    Phys(usize),
}

/// Contract the whole tree down to a single dense statevector of length `2^n`.
///
/// The output is ordered with qubit 0 as the most significant bit, matching
/// the convention in [`super::dense::DenseState`] and in `Mps::to_statevector`.
/// Intended for small `n` (≤ ~16); used by the D.2 Y-junction / star tests
/// and by `Ttn::expectation_z` in the native tree path.
pub(super) fn tree_to_statevector(sites: &[TtnSite], topology: &Topology) -> Vec<C> {
    let n = topology.n_qubits();
    assert!(n >= 1);

    // Accumulator: raw (data, shape, labels) tuple. Start with site[0].
    let mut acc_data: Vec<C> = sites[0].data.clone();
    let mut acc_shape: Vec<usize> = sites[0].shape.clone();
    let mut acc_labels: Vec<AxLabel> = sites[0]
        .edges
        .iter()
        .map(|e| AxLabel::Edge(e.0))
        .collect();
    acc_labels.push(AxLabel::Phys(0));

    let mut visited = vec![false; n];
    visited[0] = true;
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(0usize);

    while let Some(current) = queue.pop_front() {
        // Iterate over neighbours of `current` whose edge goes into the
        // accumulator (i.e. still labelled Edge(_) at this moment).
        let neighbours: Vec<EdgeId> = topology.neighbours(current).to_vec();
        for eid in neighbours {
            let edge = topology.edge(eid);
            let next = edge.other(current);
            if visited[next] {
                continue;
            }
            visited[next] = true;
            queue.push_back(next);

            // Find the accumulator axis labelled Edge(eid.0).
            let ax_acc_e = acc_labels
                .iter()
                .position(|&l| l == AxLabel::Edge(eid.0))
                .expect("outgoing edge must be present in accumulator until visited");
            let ax_next_e = sites[next].axis_for_edge(eid);

            let acc_rank = acc_shape.len();
            let rest_acc_axes: Vec<usize> = (0..acc_rank).filter(|&ax| ax != ax_acc_e).collect();
            let next_rank = sites[next].rank();
            let rest_next_axes: Vec<usize> =
                (0..next_rank).filter(|&ax| ax != ax_next_e).collect();

            // Flatten acc as [rest_acc, e] → [rows_acc, dim_e].
            let (acc_flat, rows_acc, cols_acc) =
                flatten_tensor_raw(&acc_data, &acc_shape, &rest_acc_axes, &[ax_acc_e]);
            // Flatten site[next] as [e, rest_next] → [dim_e, cols_next].
            let (next_flat, rows_next, cols_next) =
                sites[next].flatten_to_matrix(&[ax_next_e], &rest_next_axes);
            debug_assert_eq!(cols_acc, rows_next);
            let dim_e = cols_acc;

            // Matmul.
            let a_mat = Mat::from_fn(rows_acc, dim_e, |i, j| {
                let c = acc_flat[i * dim_e + j];
                faer::c64::new(c.re, c.im)
            });
            let b_mat = Mat::from_fn(dim_e, cols_next, |i, j| {
                let c = next_flat[i * cols_next + j];
                faer::c64::new(c.re, c.im)
            });
            let product = &a_mat * &b_mat;

            let mut new_data = vec![C::new(0.0, 0.0); rows_acc * cols_next];
            for i in 0..rows_acc {
                for j in 0..cols_next {
                    let z = product[(i, j)];
                    new_data[i * cols_next + j] = C::new(z.re, z.im);
                }
            }

            // New shape = rest_acc_dims ++ rest_next_dims.
            let mut new_shape: Vec<usize> = rest_acc_axes.iter().map(|&ax| acc_shape[ax]).collect();
            for &ax in &rest_next_axes {
                new_shape.push(sites[next].shape[ax]);
            }

            // New labels: rest_acc labels first, then for each rest_next axis
            // decode Edge/Phys from site[next]'s edges list and physical_axis().
            let mut new_labels: Vec<AxLabel> =
                rest_acc_axes.iter().map(|&ax| acc_labels[ax]).collect();
            let phys_next_ax = sites[next].physical_axis();
            for &ax in &rest_next_axes {
                if ax == phys_next_ax {
                    new_labels.push(AxLabel::Phys(next));
                } else {
                    // Virtual axis → look up in sites[next].edges[ax].
                    new_labels.push(AxLabel::Edge(sites[next].edges[ax].0));
                }
            }

            acc_data = new_data;
            acc_shape = new_shape;
            acc_labels = new_labels;
        }
    }

    // Permute the accumulator so the axis labels are (Phys(0), Phys(1), …, Phys(n-1)).
    let mut target_order: Vec<usize> = Vec::with_capacity(n);
    for q in 0..n {
        let pos = acc_labels
            .iter()
            .position(|&l| l == AxLabel::Phys(q))
            .expect("every qubit must be in accumulator by end of BFS");
        target_order.push(pos);
    }

    let (permuted, rows, cols) = flatten_tensor_raw(&acc_data, &acc_shape, &target_order, &[]);
    debug_assert_eq!(cols, 1);
    debug_assert_eq!(rows, 1usize << n);
    permuted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttn::topology::{Edge, Topology};

    #[test]
    fn product_state_to_statevector_y_junction() {
        // |0000⟩ on a Y-junction should give amplitude 1 at index 0 and
        // 0 everywhere else.
        let topology = Topology::from_edges(
            4,
            vec![
                Edge { a: 0, b: 1 },
                Edge { a: 0, b: 2 },
                Edge { a: 0, b: 3 },
            ],
        );
        let sites: Vec<TtnSite> = (0..4)
            .map(|v| TtnSite::product_zero(topology.neighbours(v).to_vec()))
            .collect();
        let psi = tree_to_statevector(&sites, &topology);
        assert_eq!(psi.len(), 16);
        assert!((psi[0].re - 1.0).abs() < 1e-14);
        assert!(psi[0].im.abs() < 1e-14);
        for i in 1..16 {
            assert!(psi[i].norm() < 1e-14, "nonzero amp at idx {i}: {:?}", psi[i]);
        }
    }

    #[test]
    fn product_state_to_statevector_degree_four_star() {
        // Centre 0 with 4 leaves (1,2,3,4). |00000⟩ → amplitude 1 at index 0.
        let topology = Topology::from_edges(
            5,
            vec![
                Edge { a: 0, b: 1 },
                Edge { a: 0, b: 2 },
                Edge { a: 0, b: 3 },
                Edge { a: 0, b: 4 },
            ],
        );
        let sites: Vec<TtnSite> = (0..5)
            .map(|v| TtnSite::product_zero(topology.neighbours(v).to_vec()))
            .collect();
        let psi = tree_to_statevector(&sites, &topology);
        assert_eq!(psi.len(), 32);
        assert!((psi[0].re - 1.0).abs() < 1e-14);
        for i in 1..32 {
            assert!(psi[i].norm() < 1e-14, "nonzero amp at idx {i}");
        }
    }
}
