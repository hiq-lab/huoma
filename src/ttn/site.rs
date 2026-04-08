//! Native TTN site tensor — flat `Vec<C>` storage with explicit shape and
//! edge→axis map.
//!
//! Convention: every site tensor is rank `(k + 1)` where `k` is the number of
//! incident edges. The first `k` axes are virtual legs, in the same order as
//! `edges`. The last axis is always the physical leg (dimension 2). A site
//! with zero incident edges — which only exists in the degenerate N=1 case —
//! is just a 2-vector.
//!
//! All tensor data is stored in row-major (C-order) layout: the last axis is
//! the fastest-varying. That choice puts `[virtual…, sigma]` contiguous in
//! memory for each virtual index tuple, which is the access pattern used by
//! the gate-application and observable code.
//!
//! Track D milestone D.2.

use num_complex::Complex64;

use super::topology::EdgeId;

type C = Complex64;

/// A single TTN site tensor.
#[derive(Debug, Clone)]
pub struct TtnSite {
    /// Row-major flat storage. `data.len() == shape.iter().product()`.
    pub data: Vec<C>,
    /// Shape of the tensor. Length = `edges.len() + 1`. The last entry is
    /// always `2` (the physical leg).
    pub shape: Vec<usize>,
    /// Map from virtual axis index (0..edges.len()) to the edge that axis
    /// corresponds to. Physical axis has no entry here.
    pub edges: Vec<EdgeId>,
}

impl TtnSite {
    /// Initialise to the product-|0⟩ contribution for a vertex with the
    /// given incident edges. All virtual bonds start at dimension 1.
    pub fn product_zero(edges: Vec<EdgeId>) -> Self {
        let k = edges.len();
        let mut shape = vec![1usize; k];
        shape.push(2);
        let total: usize = shape.iter().product();
        let mut data = vec![C::new(0.0, 0.0); total];
        // Only the |0⟩ amplitude is non-zero. With all virtual indices at 0
        // and sigma = 0, the flat index is also 0.
        data[0] = C::new(1.0, 0.0);
        Self { data, shape, edges }
    }

    /// Total number of axes (virtual + physical). Equals `shape.len()`.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Physical axis index. Always `shape.len() - 1`.
    pub fn physical_axis(&self) -> usize {
        self.shape.len() - 1
    }

    /// Find the axis index (into `shape`) that corresponds to the given
    /// virtual edge. Panics if the edge is not incident on this site.
    pub fn axis_for_edge(&self, e: EdgeId) -> usize {
        self.edges
            .iter()
            .position(|&x| x == e)
            .unwrap_or_else(|| panic!("edge {e:?} is not incident on this site"))
    }

    /// Dimension of the virtual axis for the given edge.
    pub fn dim_for_edge(&self, e: EdgeId) -> usize {
        self.shape[self.axis_for_edge(e)]
    }

    /// Total number of elements in the tensor.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Flatten the tensor into a matrix with `row_axes` as the row index
    /// (in the order given) and `col_axes` as the column index (in the order
    /// given). Returns `(matrix_flat, rows, cols)` with the matrix stored in
    /// row-major order.
    ///
    /// The row/col axis lists must together cover every axis of the tensor
    /// exactly once. Internally this is a `permute` followed by a reshape,
    /// but the permute is done directly into the output buffer to avoid an
    /// intermediate copy.
    pub fn flatten_to_matrix(
        &self,
        row_axes: &[usize],
        col_axes: &[usize],
    ) -> (Vec<C>, usize, usize) {
        let nd = self.rank();
        debug_assert_eq!(
            row_axes.len() + col_axes.len(),
            nd,
            "row+col axis lists must cover every axis exactly once"
        );
        // Sanity: every axis appears exactly once.
        #[cfg(debug_assertions)]
        {
            let mut seen = vec![false; nd];
            for &ax in row_axes.iter().chain(col_axes.iter()) {
                debug_assert!(ax < nd, "axis {ax} out of range");
                debug_assert!(!seen[ax], "axis {ax} appears twice in flatten_to_matrix");
                seen[ax] = true;
            }
        }

        let rows: usize = row_axes.iter().map(|&ax| self.shape[ax]).product();
        let cols: usize = col_axes.iter().map(|&ax| self.shape[ax]).product();

        // Precompute strides in the original layout (row-major).
        let mut strides = vec![0usize; nd];
        let mut acc = 1usize;
        for i in (0..nd).rev() {
            strides[i] = acc;
            acc *= self.shape[i];
        }

        // Dimensions of the new axis ordering (rows first, then cols).
        let new_order: Vec<usize> = row_axes.iter().chain(col_axes.iter()).copied().collect();
        let new_dims: Vec<usize> = new_order.iter().map(|&ax| self.shape[ax]).collect();

        let mut out = vec![C::new(0.0, 0.0); rows * cols];
        // Walk the new-layout multi-index via nested counters.
        let mut idx = vec![0usize; nd];
        let total = rows * cols;
        for flat in 0..total {
            // Compute the source offset from the new-layout index.
            let mut src = 0usize;
            for (i, &ax) in new_order.iter().enumerate() {
                src += idx[i] * strides[ax];
            }
            out[flat] = self.data[src];
            // Increment idx[nd - 1], carrying into previous axes.
            for i in (0..nd).rev() {
                idx[i] += 1;
                if idx[i] < new_dims[i] {
                    break;
                }
                idx[i] = 0;
            }
        }
        (out, rows, cols)
    }

    /// Low-level builder. Wraps an already-laid-out flat buffer in a site,
    /// performing only the lightest sanity checks. The `shape` length must
    /// equal `edges.len() + 1` (virtual legs plus the physical leg) and the
    /// data length must match the product of `shape`.
    ///
    /// This is intentionally permissive: it does **not** assert that the
    /// physical leg sits at the last axis or that the physical leg has
    /// dimension 2. Those invariants are enforced at the boundaries of the
    /// high-level operations (`product_zero`, the contraction routine, the
    /// gauge sweep) — intermediate reshapes inside those operations may
    /// transiently violate them and rely on a later permute to restore.
    pub fn unflatten_from_matrix(
        data: Vec<C>,
        new_shape: Vec<usize>,
        new_edges: Vec<EdgeId>,
    ) -> Self {
        debug_assert_eq!(
            new_shape.len(),
            new_edges.len() + 1,
            "new_shape rank must be virtual legs + 1 physical"
        );
        let total: usize = new_shape.iter().product();
        debug_assert_eq!(data.len(), total);
        Self {
            data,
            shape: new_shape,
            edges: new_edges,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_zero_single_edge() {
        let site = TtnSite::product_zero(vec![EdgeId(7)]);
        assert_eq!(site.shape, vec![1, 2]);
        assert_eq!(site.rank(), 2);
        assert_eq!(site.data.len(), 2);
        // |0⟩: sigma=0 slot is 1, sigma=1 slot is 0.
        assert_eq!(site.data[0], C::new(1.0, 0.0));
        assert_eq!(site.data[1], C::new(0.0, 0.0));
    }

    #[test]
    fn product_zero_degree_three_center() {
        let site = TtnSite::product_zero(vec![EdgeId(0), EdgeId(1), EdgeId(2)]);
        assert_eq!(site.shape, vec![1, 1, 1, 2]);
        assert_eq!(site.rank(), 4);
        assert_eq!(site.data.len(), 2);
        assert_eq!(site.data[0], C::new(1.0, 0.0));
        assert_eq!(site.data[1], C::new(0.0, 0.0));
    }

    #[test]
    fn flatten_identity_roundtrip() {
        // Build a 2x3x2 tensor with a known pattern.
        // Pretend it has two virtual edges of dim 2 and 3 plus the physical 2.
        let shape = vec![2usize, 3, 2];
        let mut data: Vec<C> = (0..12)
            .map(|i| C::new(i as f64, 0.0))
            .collect();
        // Deliberately poison with a non-trivial physical index pattern.
        for (i, d) in data.iter_mut().enumerate() {
            *d = C::new(i as f64, (i % 5) as f64);
        }
        let site = TtnSite {
            data: data.clone(),
            shape: shape.clone(),
            edges: vec![EdgeId(0), EdgeId(1)],
        };
        // Identity flatten: axes 0,1 as rows, axis 2 as cols.
        let (mat, rows, cols) = site.flatten_to_matrix(&[0, 1], &[2]);
        assert_eq!(rows, 6);
        assert_eq!(cols, 2);
        assert_eq!(mat, data);
    }

    #[test]
    fn flatten_swap_axes() {
        // 2x3 tensor, data[i*3 + j] = i*3+j.
        let data: Vec<C> = (0..6).map(|i| C::new(i as f64, 0.0)).collect();
        let shape = vec![2usize, 3];
        let site = TtnSite {
            data: data.clone(),
            shape,
            edges: vec![EdgeId(0)], // nonsense here; physical is last, but we're just testing the reshape
        };
        // Swap: axis 1 as row, axis 0 as col → 3x2 transposed.
        let (mat, rows, cols) = site.flatten_to_matrix(&[1], &[0]);
        assert_eq!(rows, 3);
        assert_eq!(cols, 2);
        // Expected transpose: [[0,3],[1,4],[2,5]] flattened row-major.
        let expected: Vec<C> = [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
            .into_iter()
            .map(|x| C::new(x, 0.0))
            .collect();
        assert_eq!(mat, expected);
    }
}
