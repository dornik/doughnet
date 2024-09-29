from typing import Literal, Optional
import numpy as np

from sdftoolbox.dual_strategies import NaiveSurfaceNetVertexStrategy, DualVertexStrategy
from sdftoolbox.types import float_dtype

class DualContouringVertexStrategy(DualVertexStrategy):
    """Computes vertex locations based on dual-contouring strategy.

    This method additionally requires intersection normals. The idea
    is to find (for each voxel) the location that agrees 'best' with
    all intersection points/normals from from surrounding active edges.

    Each interseciton point/normal consitutes a plane in 3d. A location
    agrees with it when it lies on (close-to) the plane. A location
    that agrees with all planes is considered the best.

    In practice, this can be solved by linear least squares. Consider
    an isect location p and isect normal n. Then, we'd like to find
    x, such that

        n^T(x-p) = 0

    which we can rearrange as follows

        n^Tx -n^Tp = 0
        n^Tx = n^Tp

    and concatenate (for multiple p, n pairs as) into

        Ax = b

    which we solve using least squares. However, one need to ensures that
    resulting locations lie within voxels. We do this by biasing the
    linear system to the naive SurfaceNets solution using additional
    equations of the form

        ei^Tx = bias[0]
        ej^Tx = bias[1]
        ek^Tx = bias[2]

    wher e(i,j,k) are the canonical unit vectors.

    References:
    - Ju, Tao, et al. "Dual contouring of hermite data."
    Proceedings of the 29th annual conference on Computer
    graphics and interactive techniques. 2002.
    """

    def __init__(
        self,
        bias_mode: Literal["always", "failed", "disabled"] = "always",
        bias_strength: float = 1e-5,
    ):
        """Initialize the strategy.

        Params:
            bias_mode: Determines how bias is applied. `always` applies
                biasing towards naive surface net solution unconditionally. `failed`
                applies biasing only after solving without bias results in an invalid
                point and `disable` never performs biasing. Defaults to always.
            bias_strength: Strength of bias compared to compatible normal terms.
            clip: Wether to clip the result to voxel or not. Note, when False relaxes
                the assumption that only one vertex per cell should be created. In
                particular useful when dealing with non-smooth SDFs.
        """
        assert bias_mode in ["always", "failed", "disabled"]
        self.bias_strength = bias_strength
        self.sqrt_bias_strength = np.sqrt(self.bias_strength)
        self.bias_mode = bias_mode

    def find_vertex_locations(
        self,
        active_voxels: np.ndarray,
        edge_coords: np.ndarray,
        node: "SDF",
        grid: "Grid",
    ) -> np.ndarray:
        sijk = grid.unravel_nd(active_voxels, grid.padded_shape)  # (M,3)
        active_voxel_edges = grid.find_voxel_edges(active_voxels)  # (M,12)
        points = edge_coords[active_voxel_edges]  # (M,12,3)
        # normals = node.gradient(grid.grid_to_data(points))  # original implementation -- crashes on NaNs
        # fix: grid_to_data and node.gradient crash on NaNs --> mask them and set normals to 0 elsewhere
        mask = np.isfinite(points).all(-1)
        normals = np.zeros_like(points)
        normals[mask] = node.gradient(grid.grid_to_data(points[mask]))  # (M,12,3)
        
        if self.bias_mode != "disabled":
            bias_verts = NaiveSurfaceNetVertexStrategy().find_vertex_locations(
                active_voxels, edge_coords, node, grid
            )
        else:
            bias_verts = [None] * len(points)

        # Consider a batched variant using block-diagonal matrices
        verts = []
        bias_always = self.bias_mode == "always"
        bias_failed = self.bias_mode == "failed"
        for off, p, n, bias in zip(sijk, points, normals, bias_verts):
            # off: (3,), p: (12,3), n: (12,3), bias: (3,)
            q = p - off[None, :]  # [0,1) range in each dim
            mask = np.isfinite(q).all(-1)  # Skip non-active voxel edges

            # Try to solve unbiased
            x = self._solve_lst(
                q[mask], n[mask], bias=(bias - off) if bias_always else None
            )
            if bias_failed and ((x < 0.0).any() or (x > 1.0).any()):
                x = self._solve_lst(q[mask], n[mask], bias=(bias - off))
            verts.append(x + off)
        return np.array(verts, dtype=float_dtype)

    def _solve_lst(
        self, q: np.ndarray, n: np.ndarray, bias: Optional[np.ndarray]
    ) -> np.ndarray:
        A = n
        b = (q[:, None, :] @ n[..., None]).reshape(-1)
        if bias is not None and self.bias_strength > 0.0:
            C = np.eye(3, dtype=A.dtype) * np.sqrt(self.bias_strength)
            d = bias * np.sqrt(self.bias_strength)
            A = np.concatenate((A, C), 0)
            b = np.concatenate((b, d), 0)
        x, res, rank, _ = np.linalg.lstsq(A.astype(float), b.astype(float), rcond=None)
        return x.astype(q.dtype)
