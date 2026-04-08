//! Orthogonality-center tracking and QR gauge sweeps for the native TTN.
//!
//! The gauge story in one paragraph: a TTN is in mixed canonical form when
//! all sites except a designated orthogonality center are isometries when
//! their "outgoing" (center-ward) axis is grouped as columns. In that form,
//! observables at the center are single-site quantities and truncation of
//! a bond near the center is optimal in the 2-norm sense. The center moves
//! lazily: before each two-qubit gate on edge `(a, b)`, we move the center
//! from its current location to `a`, apply the gate, and leave the new
//! center at `a`. Each move along the tree path is a QR decomposition of
//! the current-center site with its "next-step" axis as the column index —
//! the Q part stays on the current site (now an isometry with respect to
//! that axis) and the R part is contracted into the next site.
//!
//! Track D milestone D.2.

use faer::Mat;
use num_complex::Complex64;

use super::site::TtnSite;
use super::topology::{EdgeId, Topology};

type C = Complex64;

/// Move the orthogonality center from `from` to `to` via QR sweeps along the
/// unique tree path between them. After this call, every site on the path
/// except the one at `to` is an isometry with respect to its "toward-to"
/// axis, and `to` carries the entire non-isometric "bulk" of the state.
///
/// A sweep step from vertex `u` to vertex `v` along edge `e`:
///
/// 1. Reshape `site[u]` into a matrix with the `e`-axis as columns and
///    every other axis (virtual + physical) flattened into rows. Shape is
///    `[rest, dim_e]`.
/// 2. QR decompose: M = Q · R, where Q is `[rest, k]` with orthonormal
///    columns and R is `[k, dim_e]`, `k = min(rest, dim_e)`.
/// 3. Reshape Q back into a site tensor with the `e`-axis's new dimension
///    equal to `k`. `site[u]` is now isometric on the other-axes side.
/// 4. Contract R into `site[v]` along `v`'s `e`-axis. The new bond dim
///    is `k`, which replaces `dim_e` on both sites.
///
/// This never loses information (QR is exact up to floating point) and
/// never grows a bond; it can only shrink bonds that were rank-deficient.
pub fn move_center(
    sites: &mut [TtnSite],
    topology: &Topology,
    from: usize,
    to: usize,
) {
    if from == to {
        return;
    }
    let path = topology.path(from, to);
    let mut current = from;
    for edge_id in path {
        let edge = topology.edge(edge_id);
        let next = edge.other(current);
        sweep_one_step(sites, current, next, edge_id);
        current = next;
    }
    debug_assert_eq!(current, to);
}

/// Single QR sweep from `u` to `v` along edge `e`. Assumes the edge connects
/// those two vertices directly.
fn sweep_one_step(sites: &mut [TtnSite], u: usize, v: usize, e: EdgeId) {
    // Step 1: reshape site[u] into [rest, dim_e].
    let ax_u_e = sites[u].axis_for_edge(e);
    let u_rank = sites[u].rank();
    let rest_axes_u: Vec<usize> = (0..u_rank).filter(|&ax| ax != ax_u_e).collect();
    let (mat_flat, rows, cols) = sites[u].flatten_to_matrix(&rest_axes_u, &[ax_u_e]);
    let dim_e_old = cols;

    // Step 2: QR.
    let mat = Mat::from_fn(rows, cols, |i, j| {
        let c = mat_flat[i * cols + j];
        faer::c64::new(c.re, c.im)
    });
    let qr = mat.qr();
    let q_mat = qr.compute_thin_Q();
    let r_mat = qr.thin_R();
    let k = q_mat.ncols();
    debug_assert_eq!(k, rows.min(cols));
    debug_assert_eq!(r_mat.nrows(), k);
    debug_assert_eq!(r_mat.ncols(), cols);

    // Step 3: rebuild site[u] from Q. Q has shape [rest_flat, k]. The new
    // axis ordering is [rest_axes_u (in original order), e-axis with dim k].
    let mut q_flat: Vec<C> = Vec::with_capacity(rows * k);
    for i in 0..rows {
        for j in 0..k {
            let z = q_mat[(i, j)];
            q_flat.push(C::new(z.re, z.im));
        }
    }
    // new_shape for the [rest, e] layout.
    let mut new_shape_u: Vec<usize> = rest_axes_u
        .iter()
        .map(|&ax| sites[u].shape[ax])
        .collect();
    new_shape_u.push(k);
    // We also need the physical axis to remain in its position in the
    // concatenation. Since physical lives at rank-1 in the old layout and
    // we put it into rest_axes_u (which excludes only ax_u_e), the physical
    // axis may sit *anywhere* in the new layout depending on where
    // ax_u_e was. We need to canonicalise: after this sweep, the site must
    // still satisfy the "last axis is physical" invariant.
    let physical_in_new = rest_axes_u
        .iter()
        .position(|&ax| ax == sites[u].physical_axis())
        .expect("physical axis should be in rest_axes_u since it can't be the e-axis");
    // `new_shape_u` currently has `k` at the end and physical at index
    // `physical_in_new`. We need physical at the end. Permute by swapping
    // the physical slot with the last virtual-edge slot (at index nd-2),
    // *but* only if they are not already in order. Actually we need the
    // "last axis is physical" invariant. After appending `k`, the shape is
    // [..., physical_in_new=2, ..., k]. We want [..., k, ..., 2].
    //
    // Simpler approach: do a full permute to put physical at the very end.
    let built = TtnSite::unflatten_from_matrix(
        q_flat,
        new_shape_u.clone(),
        // temporary edges list matching the current layout: rest virtual
        // edges then the e-edge. We'll fix the physical-last invariant via
        // a flatten_to_matrix reshape below.
        {
            let mut tmp = Vec::with_capacity(new_shape_u.len() - 1);
            for (i, &ax) in rest_axes_u.iter().enumerate() {
                if ax == sites[u].physical_axis() {
                    continue; // skip physical; not an edge
                }
                let _ = i;
                tmp.push(sites[u].edges[ax]);
            }
            tmp.push(e);
            tmp
        },
    );
    // Now permute so physical is last.
    // The current layout is [rest_axes_u in original order, e].
    // Physical sits at position `physical_in_new`. We want it at the end.
    // Build a final-order axis list that is:
    //   [everything except physical in current order, then physical]
    let nd_built = built.rank();
    let phys_current = physical_in_new; // where physical is in `built`
    let mut final_order: Vec<usize> = (0..nd_built).filter(|&i| i != phys_current).collect();
    final_order.push(phys_current);
    // If it's already in order, skip the reshape.
    let needs_permute: bool = final_order
        .iter()
        .enumerate()
        .any(|(i, &j)| i != j);
    let site_u_new = if needs_permute {
        // Use flatten_to_matrix to execute the permutation. Everything goes
        // into the row side so the "cols = 1" degenerate case is harmless.
        let (permuted, tot, one) = built.flatten_to_matrix(&final_order, &[]);
        debug_assert_eq!(one, 1);
        debug_assert_eq!(tot, built.len());
        let final_shape: Vec<usize> = final_order.iter().map(|&i| built.shape[i]).collect();
        let final_edges: Vec<EdgeId> = final_order
            .iter()
            .filter(|&&i| i != phys_current)
            .map(|&i| {
                // built.edges has length nd_built - 1; indices 0..nd_built-1
                // correspond to axes 0..nd_built-1 (axis nd_built-1 is
                // physical). But here physical is at phys_current, not at
                // the end. We need to map axis index → edge, skipping
                // physical.
                if i < phys_current {
                    built.edges[i]
                } else {
                    // axes after phys_current in `built` map to built.edges
                    // index (i - 1) because built.edges skips the physical
                    // slot. Wait — built.edges was constructed as
                    // [rest virtual edges, e], and `built.shape` has
                    // physical at position phys_current. So built.edges
                    // indexes by "virtual axis slot", not by raw axis index.
                    // The virtual axis slots are 0..nd_built - 1 with
                    // phys_current removed.
                    built.edges[i - 1]
                }
            })
            .collect();
        TtnSite::unflatten_from_matrix(permuted, final_shape, final_edges)
    } else {
        built
    };

    sites[u] = site_u_new;

    // Step 4: contract R into site[v] along v's e-axis.
    let ax_v_e = sites[v].axis_for_edge(e);
    // site[v] has shape [..., dim_e_old, ...]. We want to replace that axis
    // with the multiplication by R.T (R is [k, dim_e_old], and contracting
    // R into site[v] along dim_e_old gives [k, rest_of_v]).
    //
    // Flatten site[v] with its e-axis as ROWS (so it becomes
    // [dim_e_old, rest_flat]) and multiply R [k, dim_e_old] on the left.
    let v_rank = sites[v].rank();
    let rest_axes_v: Vec<usize> = (0..v_rank).filter(|&ax| ax != ax_v_e).collect();
    let (v_flat, v_rows, v_cols) = sites[v].flatten_to_matrix(&[ax_v_e], &rest_axes_v);
    debug_assert_eq!(v_rows, dim_e_old);
    let v_mat = Mat::from_fn(v_rows, v_cols, |i, j| {
        let c = v_flat[i * v_cols + j];
        faer::c64::new(c.re, c.im)
    });
    // Result: R_mat (k × dim_e_old) · v_mat (dim_e_old × v_cols) = (k × v_cols).
    let r_owned: Mat<faer::c64> = Mat::from_fn(k, cols, |i, j| r_mat[(i, j)]);
    let product = &r_owned * &v_mat;
    let prod_rows = product.nrows();
    let prod_cols = product.ncols();
    debug_assert_eq!(prod_rows, k);
    debug_assert_eq!(prod_cols, v_cols);
    let mut prod_flat: Vec<C> = Vec::with_capacity(prod_rows * prod_cols);
    for i in 0..prod_rows {
        for j in 0..prod_cols {
            let z = product[(i, j)];
            prod_flat.push(C::new(z.re, z.im));
        }
    }
    // Now unflatten: the result is in layout [e-axis first (dim k), rest_of_v].
    // That matches the current "e-axis at position 0" ordering, so the
    // shape is [k, rest_dims...]. We need to permute back so the edges
    // appear in site[v]'s original axis order, with physical still last.
    //
    // Build the [e-first] new shape and permute into v's original order.
    let mut intermediate_shape: Vec<usize> = Vec::with_capacity(v_rank);
    intermediate_shape.push(k);
    for &ax in &rest_axes_v {
        intermediate_shape.push(sites[v].shape[ax]);
    }
    let mut intermediate_edges: Vec<EdgeId> = Vec::new();
    intermediate_edges.push(e);
    for &ax in &rest_axes_v {
        if ax == sites[v].physical_axis() {
            continue;
        }
        intermediate_edges.push(sites[v].edges[ax]);
    }
    let intermediate = TtnSite::unflatten_from_matrix(
        prod_flat,
        intermediate_shape.clone(),
        intermediate_edges,
    );

    // Permute the intermediate back to the original v axis order. In the
    // intermediate, axis 0 is the e-axis and axes 1..nd are rest_axes_v in
    // order. We want the final axis order to be v's original order (i.e.,
    // for each original axis `ax` in 0..v_rank, pick it from the
    // intermediate layout). Build the permutation:
    //   for each original axis position o:
    //     if o == ax_v_e: pick intermediate axis 0
    //     else: pick intermediate axis (rest_axes_v.position(o) + 1)
    let mut target_order: Vec<usize> = Vec::with_capacity(v_rank);
    for o in 0..v_rank {
        if o == ax_v_e {
            target_order.push(0);
        } else {
            let pos = rest_axes_v.iter().position(|&x| x == o).expect("rest_axes_v covers all non-e axes");
            target_order.push(pos + 1);
        }
    }
    let needs_permute_v = target_order
        .iter()
        .enumerate()
        .any(|(i, &j)| i != j);
    let site_v_new = if needs_permute_v {
        let (permuted, tot, one) = intermediate.flatten_to_matrix(&target_order, &[]);
        debug_assert_eq!(one, 1);
        debug_assert_eq!(tot, intermediate.len());
        let final_shape: Vec<usize> = target_order
            .iter()
            .map(|&i| intermediate.shape[i])
            .collect();
        // Edges in final order: skip the physical axis in v's original
        // order.
        let mut final_edges: Vec<EdgeId> = Vec::with_capacity(v_rank - 1);
        let phys_v = sites[v].physical_axis();
        for o in 0..v_rank {
            if o == phys_v {
                continue;
            }
            if o == ax_v_e {
                final_edges.push(e);
            } else {
                final_edges.push(sites[v].edges[o]);
            }
        }
        TtnSite::unflatten_from_matrix(permuted, final_shape, final_edges)
    } else {
        intermediate
    };

    sites[v] = site_v_new;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttn::topology::{Edge, Topology};

    #[test]
    fn move_center_noop_same_site() {
        let topology = Topology::linear_chain(3);
        let mut sites: Vec<TtnSite> = (0..3)
            .map(|v| TtnSite::product_zero(topology.neighbours(v).to_vec()))
            .collect();
        // Snapshot before.
        let before: Vec<Vec<C>> = sites.iter().map(|s| s.data.clone()).collect();
        move_center(&mut sites, &topology, 1, 1);
        // Nothing should change.
        for (i, s) in sites.iter().enumerate() {
            assert_eq!(s.data, before[i]);
        }
    }

    #[test]
    fn move_center_preserves_norm_on_product_state() {
        // Y-junction, product |0000⟩ → moving the center around should
        // leave the center tensor equal to |0⟩ (the isometric leaves are
        // all trivial identities since initial bond dim = 1).
        let topology = Topology::from_edges(
            4,
            vec![
                Edge { a: 0, b: 1 },
                Edge { a: 0, b: 2 },
                Edge { a: 0, b: 3 },
            ],
        );
        let mut sites: Vec<TtnSite> = (0..4)
            .map(|v| TtnSite::product_zero(topology.neighbours(v).to_vec()))
            .collect();
        // Sanity: the center is at some implicit starting vertex (caller's
        // choice); moving to any other vertex should still leave every
        // tensor with total-norm 1 (product state, no truncation).
        move_center(&mut sites, &topology, 0, 2);
        for s in &sites {
            let norm: f64 = s.data.iter().map(|c| c.norm_sqr()).sum();
            assert!(
                (norm - 1.0).abs() < 1e-12,
                "site norm drifted after gauge move: got {norm}"
            );
        }
    }
}
