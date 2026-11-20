import numpy as np
import pathlib
import imageio
import torch
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from skimage.morphology import skeletonize, skeletonize_3d
from skimage.measure import label as cc_label
from scipy.ndimage import binary_dilation, convolve
from collections import defaultdict
from scipy.spatial import cKDTree
from precomputed_tif.client import ArrayReader


def pretif_open(p):
    """
    p: path to pretif folder
    """
    return ArrayReader(pathlib.Path(p).as_uri(), format="blockfs")


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


def match_cell(centroids, ground_truth, R=5):
    if isinstance(centroids, np.ndarray):
        centroids = centroids.tolist()
    if isinstance(ground_truth, np.ndarray) or isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.tolist()
    gt_copy = ground_truth.copy()
    centroids_copy = centroids.copy()
    tp_list = []
    fn_list = []

    while len(gt_copy) != 0 and len(centroids_copy) != 0:
        point = gt_copy.pop()
        dist = np.linalg.norm(np.array(centroids_copy) - point, axis=1)
        idx = np.argmin(dist)
        max_dist = dist[idx]
        if max_dist <= R:
            centroids_copy.pop(idx)
            tp_list.append(point)
        else:
            fn_list.append(point)
    fp_list = centroids_copy
    if len(gt_copy) > 0:
        fn_list += gt_copy

    return tp_list, fn_list, fp_list


def evaluate_metrics(tp, fn, fp):
    if tp + fp == 0:
        precision = 0
        print("Warning: precision is set to 0 due to no positive predictions.")
    else:
        precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = tp / (tp + fn + fp)

    return precision, recall, accuracy


def _voxel_rasterize_polyline(points, steps_dict=None):
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return pts

    voxel_pts = np.rint(pts).astype(int, copy=False)
    if voxel_pts.shape[0] < 2:
        return voxel_pts.astype(float, copy=False)

    segments = []
    for start, end in zip(voxel_pts[:-1], voxel_pts[1:]):
        delta = end - start
        step = int(np.max(np.abs(delta)))
        if steps_dict is not None:
            steps_dict[step] += 1
        if step <= 0:
            seg = start.reshape(1, -1)
        else:
            t = np.linspace(0.0, 1.0, step + 1, dtype=float)
            seg = np.rint(start + delta * t[:, None]).astype(int, copy=False)
        if segments and np.array_equal(segments[-1][-1], seg[0]):
            seg = seg[1:]
        if seg.size:
            segments.append(seg)

    if not segments:
        return voxel_pts[:1].astype(float, copy=False)
    out = np.concatenate(segments, axis=0)
    if out.shape[0] > 1:
        keep = np.ones((out.shape[0],), dtype=bool)
        keep[1:] = np.any(out[1:] != out[:-1], axis=1)
        out = out[keep]
    return out.astype(float, copy=False)


def connect_neurite_points(points, max_step=0):
    """Rasterize a polyline to voxel coordinates with 26-connected consecutive points.

    max_step is kept for API compatibility; rasterization now always fills voxel gaps.
    """
    return _voxel_rasterize_polyline(points)


def build_skeleton_graph(pts, connectivity=26):
    pts = np.asarray(pts, dtype=int)
    N = pts.shape[0]
    if N == 0:
        return pts, {}, np.zeros((0,), dtype=int)

    coord_to_idx = {tuple(p): i for i, p in enumerate(pts)}

    if connectivity == 6:
        offsets = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]
    elif connectivity == 26:
        offsets = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    offsets.append((dz, dy, dx))
    else:
        raise ValueError("connectivity must be 6 or 26")

    adj = defaultdict(list)
    for i, (z, y, x) in enumerate(pts):
        for dz, dy, dx in offsets:
            j = coord_to_idx.get((z + dz, y + dy, x + dx))
            if j is not None:
                adj[i].append(j)

    for i in range(N):
        _ = adj[i]

    degrees = np.array([len(adj[i]) for i in range(N)], dtype=int)
    return pts, adj, degrees


def prune_short_leaf_branches(pts, adj, spur_len=8):
    pts = pts.copy()
    N = pts.shape[0]
    if N == 0:
        return pts, adj

    nbrs = [set(adj[i]) for i in range(N)]

    def degree(i):
        return len(nbrs[i])

    removed = np.zeros(N, dtype=bool)
    changed = True

    while changed:
        changed = False
        deg = np.array([degree(i) if not removed[i] else 0 for i in range(N)], dtype=int)
        leaves = np.where((deg == 1) & (~removed))[0]
        if leaves.size == 0:
            break

        for leaf in leaves:
            if removed[leaf] or degree(leaf) != 1:
                continue

            path = [leaf]
            cur = leaf
            prev = -1

            while True:
                if degree(cur) == 0:
                    break
                if cur != leaf and degree(cur) != 2:
                    break
                nxt_candidates = [n for n in nbrs[cur] if n != prev and not removed[n]]
                if not nxt_candidates:
                    break
                nxt = nxt_candidates[0]
                prev, cur = cur, nxt
                path.append(cur)

            if len(path) <= spur_len:
                keep_last = degree(path[-1]) >= 3
                to_remove = path[:-1] if keep_last else path
                if to_remove:
                    changed = True
                    for u in to_remove:
                        removed[u] = True
                        for v in list(nbrs[u]):
                            nbrs[v].discard(u)
                        nbrs[u].clear()

    keep = np.where(~removed)[0]
    if keep.size == 0:
        return np.zeros((0, 3), dtype=int), defaultdict(list)

    old2new = {old: new for new, old in enumerate(keep)}
    pts2 = pts[keep]
    adj2 = defaultdict(list)
    for old_u in keep:
        u = old2new[old_u]
        for old_v in nbrs[old_u]:
            if old_v in old2new:
                adj2[u].append(old2new[old_v])
    for i in range(pts2.shape[0]):
        _ = adj2[i]
    return pts2, adj2


def extract_branch_paths(pts, adj):
    N = pts.shape[0]
    deg = np.array([len(adj[i]) for i in range(N)], dtype=int)
    junction = deg != 2

    used = set()

    def mark_edge(u, v):
        if u < v:
            used.add((u, v))
        else:
            used.add((v, u))

    def edge_used(u, v):
        return (u, v) in used if u < v else (v, u) in used

    paths = []

    for s in range(N):
        if not junction[s]:
            continue
        for n in adj[s]:
            if edge_used(s, n):
                continue

            nodes = [s]
            prev = s
            cur = n
            mark_edge(s, n)

            while True:
                nodes.append(cur)
                if junction[cur]:
                    paths.append({"start": s, "end": cur, "nodes": nodes})
                    break
                nbrs = adj[cur]
                nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
                if edge_used(cur, nxt):
                    paths.append({"start": s, "end": cur, "nodes": nodes})
                    break
                mark_edge(cur, nxt)
                prev, cur = cur, nxt

    loops = []
    if not np.any(junction) and N > 0:
        visited = np.zeros(N, dtype=bool)
        for start in range(N):
            if visited[start]:
                continue
            cycle = [start]
            visited[start] = True
            prev = -1
            cur = start
            while True:
                nbrs = adj[cur]
                nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
                if nxt == start:
                    cycle.append(start)
                    break
                if visited[nxt]:
                    break
                visited[nxt] = True
                cycle.append(nxt)
                prev, cur = cur, nxt
            loops.append(cycle)

    return junction, paths, loops


def smooth_polyline(points, win=5):
    pts = np.asarray(points, dtype=float)
    N = pts.shape[0]
    if N == 0 or win <= 1 or N < win:
        return pts
    win = int(win)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    padded = np.pad(pts, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(pts)
    for i in range(N):
        out[i] = padded[i : i + win].mean(axis=0)
    return out


def simplify_path_with_fixed_endpoints(path_pts, smooth_win=5, rdp_eps=1.5):
    if path_pts.shape[0] <= 2:
        return path_pts.astype(float, copy=False)

    p0 = path_pts[0].astype(float)
    p1 = path_pts[-1].astype(float)

    sm = smooth_polyline(path_pts, win=smooth_win)
    sm[0] = p0
    sm[-1] = p1
    return sm  # RDP is too aggressive and causing issues, return the smoothed path only for now
    # # TODO: RDP is too aggressive and causing issues
    # simp[0] = p0
    # simp[-1] = p1


def skeleton_to_simplified_paths(
    skel_pts,
    connectivity=26,
    keep_largest_cc=True,
    spur_len=0,
    smooth_win=5,
    rdp_eps=1.5,
    max_step=0,
):
    def _draw_line_local(p0, p1):
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        delta = p1 - p0
        steps = int(np.max(np.abs(delta)))
        if steps <= 0:
            return np.rint(p0[None, :]).astype(int)
        t = np.linspace(0.0, 1.0, steps + 1, dtype=float)
        pts = p0[None, :] + t[:, None] * delta[None, :]
        pts = np.rint(pts).astype(int)
        if pts.shape[0] <= 1:
            return pts
        keep = [0]
        for i in range(1, pts.shape[0]):
            if not np.all(pts[i] == pts[keep[-1]]):
                keep.append(i)
        return pts[np.array(keep, dtype=int)]

    def _merge_endpoints_within_step(coords, step):
        coords = np.asarray(coords, dtype=int)
        if coords.size == 0 or step <= 0:
            return coords
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        start = mins - 1
        end = maxs + 2
        shape = (end - start).astype(int)
        if np.any(shape <= 0):
            return coords
        local = coords - start
        valid = np.all((local >= 0) & (local < shape), axis=1)
        local = local[valid]
        if local.size == 0:
            return coords
        mask = np.zeros(tuple(shape.tolist()), dtype=bool)
        mask[tuple(local.T)] = True

        kernel = np.ones((3,) * mask.ndim, dtype=np.uint8)
        kernel[(1,) * mask.ndim] = 0
        deg = convolve(mask.astype(np.uint8), kernel, mode="constant", cval=0)
        endpoints = np.argwhere(mask & (deg == 1))
        if endpoints.shape[0] < 2:
            return coords

        comp = cc_label(mask.astype(np.uint8), connectivity=mask.ndim)
        comp_ids = comp[tuple(endpoints.T)]
        if len(np.unique(comp_ids)) <= 1:
            return coords

        ep_world = endpoints + start
        tree = cKDTree(ep_world.astype(float))
        parent = {int(c): int(c) for c in np.unique(comp_ids) if c > 0}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        for i, p in enumerate(ep_world):
            nbrs = tree.query_ball_point(p, r=float(step))
            for j in nbrs:
                if j <= i:
                    continue
                c1 = int(comp_ids[i])
                c2 = int(comp_ids[j])
                if c1 <= 0 or c2 <= 0:
                    continue
                if find(c1) == find(c2):
                    continue
                line_pts = _draw_line_local(ep_world[i], ep_world[j])
                local_line = line_pts - start
                inb = np.all((local_line >= 0) & (local_line < shape), axis=1)
                local_line = local_line[inb]
                if local_line.size:
                    mask[tuple(local_line.T)] = True
                union(c1, c2)

        merged = np.argwhere(mask) + start
        return merged.astype(int)

    skel_pts = np.asarray(skel_pts, dtype=int)
    if skel_pts.size == 0:
        return {
            "junction_pts": np.zeros((0, 3)),
            "junction_ids": np.zeros((0,), dtype=int),
            "branches": [],
            "loops": [],
        }

    if max_step and max_step > 0:
        skel_pts = _merge_endpoints_within_step(skel_pts, max_step)

    pts, adj, _ = build_skeleton_graph(skel_pts, connectivity=connectivity)

    # missing some long branches in this step, skipped
    #     pts, adj = largest_connected_component(pts, adj)

    if spur_len and spur_len > 0:
        pts, adj = prune_short_leaf_branches(pts, adj, spur_len=int(spur_len))

    junction_mask, paths, loops = extract_branch_paths(pts, adj)
    junction_ids = np.where(junction_mask)[0]
    junction_pts = pts[junction_ids].astype(float)

    jmap = {node: j for j, node in enumerate(junction_ids.tolist())}

    branches = []
    for p in paths:
        s = p["start"]
        e = p["end"]
        if s not in jmap or e not in jmap:
            continue
        node_list = p["nodes"]
        path_pts = pts[np.array(node_list, dtype=int)]
        poly = simplify_path_with_fixed_endpoints(path_pts, smooth_win=smooth_win, rdp_eps=rdp_eps)
        branches.append(
            {"start_junction": jmap[s], "end_junction": jmap[e], "polyline": poly,}
        )

    loop_out = []
    for cyc in loops:
        cyc_pts = pts[np.array(cyc, dtype=int)]
        poly = simplify_path_with_fixed_endpoints(cyc_pts, smooth_win=smooth_win, rdp_eps=rdp_eps)
        loop_out.append({"polyline": poly})

    return {
        "junction_pts": junction_pts,
        "junction_ids": junction_ids,
        "branches": branches,
        "loops": loop_out,
    }


# ---------------------------
# Crossing split + clean (2D/3D)
# ---------------------------

_NEIGHBOR_OFFSETS = {}
_NEIGHBOR_KERNELS = {}


def _get_neighbor_offsets(ndim):
    offsets = _NEIGHBOR_OFFSETS.get(ndim)
    if offsets is not None:
        return offsets
    offsets = []
    for delta in np.ndindex(*(3,) * ndim):
        d = tuple(int(v) - 1 for v in delta)
        if all(v == 0 for v in d):
            continue
        offsets.append(d)
    _NEIGHBOR_OFFSETS[ndim] = offsets
    return offsets


def _get_neighbor_kernel(ndim):
    kernel = _NEIGHBOR_KERNELS.get(ndim)
    if kernel is not None:
        return kernel
    shape = (3,) * ndim
    kernel = np.ones(shape, dtype=np.uint8)
    center = (1,) * ndim
    kernel[center] = 0
    _NEIGHBOR_KERNELS[ndim] = kernel
    return kernel


def _neighbor_count_nd(skel: np.ndarray) -> np.ndarray:
    kernel = _get_neighbor_kernel(skel.ndim)
    return convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u) + 1e-8
    nv = np.linalg.norm(v) + 1e-8
    c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _estimate_dir_from_path(path_pts: List[Tuple[int, ...]], k_dir: int) -> np.ndarray:
    if len(path_pts) < 2:
        return np.zeros((len(path_pts[0]) if path_pts else 3,), dtype=np.float32)
    k = min(k_dir, len(path_pts) - 1)
    p0 = np.asarray(path_pts[0], dtype=np.float32)
    p1 = np.asarray(path_pts[k], dtype=np.float32)
    v = p1 - p0
    n = np.linalg.norm(v) + 1e-8
    return (v / n).astype(np.float32)


def _iter_neighbors(coord, shape, offsets):
    for d in offsets:
        nbr = tuple(coord[i] + d[i] for i in range(len(d)))
        in_bounds = True
        for i, v in enumerate(nbr):
            if v < 0 or v >= shape[i]:
                in_bounds = False
                break
        if in_bounds:
            yield nbr


@dataclass
class _EdgeTrace:
    start_node: int
    end_node: int
    start_boundary: Tuple[int, ...]
    end_boundary: Tuple[int, ...]
    path: List[Tuple[int, ...]]
    length: int
    dir_from_start: np.ndarray


def _build_supernodes(skel: np.ndarray, deg: np.ndarray) -> Tuple[np.ndarray, int]:
    node_mask = skel & (deg != 2)
    node_id_map = cc_label(node_mask.astype(np.uint8), connectivity=skel.ndim)
    return node_id_map, int(node_id_map.max())


def _trace_edge_from_boundary(
    skel: np.ndarray,
    deg: np.ndarray,
    node_id_map: np.ndarray,
    start_node_id: int,
    start_boundary: Tuple[int, ...],
    k_dir: int,
    offsets: List[Tuple[int, ...]],
    max_steps: int = 200000,
) -> Optional[_EdgeTrace]:
    if not skel[start_boundary] or node_id_map[start_boundary] != 0:
        return None

    path = [start_boundary]
    prev = None
    cur = start_boundary

    for _ in range(max_steps):
        nbrs = []
        for nbr in _iter_neighbors(cur, skel.shape, offsets):
            if not skel[nbr]:
                continue
            if prev is not None and nbr == prev:
                continue
            nbrs.append(nbr)

        if not nbrs:
            break

        nxt = nbrs[0]
        prev = cur
        cur = nxt

        end_node_id = int(node_id_map[cur])
        if end_node_id != 0:
            if end_node_id == start_node_id:
                return None
            end_boundary = prev
            dir_vec = _estimate_dir_from_path(path, k_dir=k_dir)
            return _EdgeTrace(
                start_node=start_node_id,
                end_node=end_node_id,
                start_boundary=start_boundary,
                end_boundary=end_boundary,
                path=path,
                length=len(path),
                dir_from_start=dir_vec,
            )

        path.append(cur)

        if deg[cur] != 2 and node_id_map[cur] == 0:
            break

    return None


def _collect_edges(
    skel: np.ndarray, deg: np.ndarray, node_id_map: np.ndarray, num_nodes: int, k_dir: int,
) -> Tuple[List[_EdgeTrace], Dict[int, List[int]]]:
    edges: List[_EdgeTrace] = []
    node_to_edges: Dict[int, List[int]] = {i: [] for i in range(1, num_nodes + 1)}
    offsets = _get_neighbor_offsets(skel.ndim)

    visited_boundary = set()
    seen_edges = set()

    for node_id in range(1, num_nodes + 1):
        node_pixels = np.argwhere(node_id_map == node_id)
        node_set = set(map(tuple, node_pixels.tolist()))

        boundary = set()
        for coord in node_set:
            for nbr in _iter_neighbors(coord, skel.shape, offsets):
                if skel[nbr] and nbr not in node_set and node_id_map[nbr] == 0:
                    boundary.add(nbr)

        for b in boundary:
            key = (node_id, b)
            if key in visited_boundary:
                continue
            visited_boundary.add(key)

            et = _trace_edge_from_boundary(
                skel=skel,
                deg=deg,
                node_id_map=node_id_map,
                start_node_id=node_id,
                start_boundary=b,
                k_dir=k_dir,
                offsets=offsets,
            )
            if et is None:
                continue

            sig_nodes = (min(et.start_node, et.end_node), max(et.start_node, et.end_node))
            sig_bounds = tuple(sorted([et.start_boundary, et.end_boundary]))
            sig = (sig_nodes, sig_bounds)
            if sig in seen_edges:
                continue
            seen_edges.add(sig)

            edges.append(et)
            idx = len(edges) - 1
            node_to_edges[et.start_node].append(idx)
            node_to_edges[et.end_node].append(idx)

    return edges, node_to_edges


def _is_crossing_like(incident_dirs: List[np.ndarray], theta_straight: float = 150.0) -> bool:
    m = len(incident_dirs)
    if m < 4:
        return False

    dirs = []
    for d in incident_dirs:
        n = np.linalg.norm(d) + 1e-8
        dirs.append((d / n).astype(np.float32))

    pairs = []
    for i in range(m):
        for j in range(i + 1, m):
            a = _angle_deg(dirs[i], dirs[j])
            if a >= theta_straight:
                pairs.append((i, j, a))

    if len(pairs) < 2:
        return False

    for p1 in pairs:
        for p2 in pairs:
            if p1 is p2:
                continue
            s = {p1[0], p1[1], p2[0], p2[1]}
            if len(s) == 4:
                return True

    return False


def _find_best_cross_pairs(
    incident_dirs: List[np.ndarray], theta_straight: float, lengths: Optional[List[int]] = None,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    m = len(incident_dirs)
    if m < 4:
        return None

    dirs = []
    for d in incident_dirs:
        n = np.linalg.norm(d) + 1e-8
        dirs.append((d / n).astype(np.float32))

    pairs = []
    for i in range(m):
        for j in range(i + 1, m):
            a = _angle_deg(dirs[i], dirs[j])
            if a >= theta_straight:
                pairs.append((i, j, a))

    if len(pairs) < 2:
        return None

    best = None
    best_score = -1e9
    best_len = -1e9
    for p1 in pairs:
        for p2 in pairs:
            if p1 is p2:
                continue
            s = {p1[0], p1[1], p2[0], p2[1]}
            if len(s) != 4:
                continue
            score = p1[2] + p2[2]
            len_score = 0.0
            if lengths is not None:
                len_score = lengths[p1[0]] + lengths[p1[1]] + lengths[p2[0]] + lengths[p2[1]]
            if score > best_score or (score == best_score and len_score > best_len):
                best_score = score
                best_len = len_score
                best = ((p1[0], p1[1]), (p2[0], p2[1]))

    return best


def _edge_component_id(
    edge: _EdgeTrace, node_id: int, comp: np.ndarray, sk_cut: np.ndarray,
) -> int:
    if edge.start_node == node_id:
        path = edge.path
    else:
        path = list(reversed(edge.path))

    for coord in path:
        if sk_cut[coord]:
            cid = int(comp[coord])
            if cid > 0:
                return cid
    return 0


def _component_endpoints(mask: np.ndarray) -> np.ndarray:
    deg = _neighbor_count_nd(mask)
    endpoints = np.argwhere(mask & (deg == 1))
    return endpoints


def _closest_endpoint_pair(
    a: np.ndarray, b: np.ndarray
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    if a.size == 0 or b.size == 0:
        return float("inf"), None, None
    a = a.astype(float)
    b = b.astype(float)
    # brute force is fine for endpoint counts
    d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    idx = np.unravel_index(np.argmin(d2), d2.shape)
    dist = float(np.sqrt(d2[idx]))
    return dist, a[idx[0]], b[idx[1]]


def _prune_short_spurs_simple(skel: np.ndarray, L_prune: int, max_iters: int = 50) -> np.ndarray:
    skel = skel.copy().astype(bool)
    offsets = _get_neighbor_offsets(skel.ndim)

    for _ in range(max_iters):
        deg = _neighbor_count_nd(skel)
        node_mask = skel & (deg != 2)
        endpoint_mask = skel & (deg == 1)
        endpoints = np.argwhere(endpoint_mask)
        if endpoints.size == 0:
            break

        to_remove = []
        for ep in endpoints:
            start = tuple(int(v) for v in ep)
            cur = start
            prev = None
            path = [cur]

            for _step in range(100000):
                nbrs = []
                for nbr in _iter_neighbors(cur, skel.shape, offsets):
                    if not skel[nbr]:
                        continue
                    if prev is not None and nbr == prev:
                        continue
                    nbrs.append(nbr)
                if not nbrs:
                    break
                nxt = nbrs[0]
                prev = cur
                cur = nxt
                path.append(cur)
                if node_mask[cur] and cur != start:
                    break

            if len(path) <= L_prune and node_mask[path[-1]]:
                to_remove.extend(path[:-1])

        if not to_remove:
            break

        coords = np.array(to_remove, dtype=int)
        skel[tuple(coords.T)] = False

    return skel


def _draw_line_nd(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    delta = p1 - p0
    steps = int(np.max(np.abs(delta)))
    if steps <= 0:
        return np.rint(p0[None, :]).astype(int)
    t = np.linspace(0.0, 1.0, steps + 1, dtype=float)
    pts = p0[None, :] + t[:, None] * delta[None, :]
    pts = np.rint(pts).astype(int)
    if pts.shape[0] <= 1:
        return pts
    # remove consecutive duplicates
    keep = [0]
    for i in range(1, pts.shape[0]):
        if not np.all(pts[i] == pts[keep[-1]]):
            keep.append(i)
    return pts[np.array(keep, dtype=int)]


def _bridge_small_gaps(
    skel: np.ndarray, R_bridge: int = 6, cos_min: float = 0.7, k_dir: int = 6,
) -> np.ndarray:
    skel = skel.copy().astype(bool)
    deg = _neighbor_count_nd(skel)
    endpoints = np.argwhere(skel & (deg == 1))
    if endpoints.shape[0] < 2:
        return skel

    offsets = _get_neighbor_offsets(skel.ndim)

    def inward_path_from_endpoint(ep):
        cur = tuple(int(v) for v in ep)
        path = [cur]
        prev = None
        for _ in range(200):
            nbrs = []
            for nbr in _iter_neighbors(cur, skel.shape, offsets):
                if not skel[nbr]:
                    continue
                if prev is not None and nbr == prev:
                    continue
                nbrs.append(nbr)
            if not nbrs:
                break
            nxt = nbrs[0]
            prev = cur
            cur = nxt
            path.append(cur)
            if len(path) >= k_dir + 1:
                break
        return path

    coords = endpoints.astype(np.float32)
    dirs = []
    for ep in endpoints:
        p = inward_path_from_endpoint(ep)
        d = _estimate_dir_from_path(p, k_dir=min(k_dir, len(p) - 1))
        dirs.append(d)
    dirs = np.array(dirs, dtype=np.float32)

    tree = cKDTree(coords)
    used = np.zeros(len(coords), dtype=bool)

    for i in range(len(coords)):
        if used[i]:
            continue
        nbrs = tree.query_ball_point(coords[i], r=R_bridge)
        best_j = None
        best_score = -1e9

        for j in nbrs:
            if j == i or used[j]:
                continue
            dvec = coords[j] - coords[i]
            dist = np.linalg.norm(dvec) + 1e-8
            u = dvec / dist
            score_i = float(np.dot(dirs[i], -u))
            score_j = float(np.dot(dirs[j], u))
            score = 0.5 * (score_i + score_j)
            if score >= cos_min:
                combo = score - 0.05 * dist
                if combo > best_score:
                    best_score = combo
                    best_j = j

        if best_j is not None:
            used[i] = True
            used[best_j] = True
            p0 = coords[i].astype(int)
            p1 = coords[best_j].astype(int)
            line_pts = _draw_line_nd(p0, p1)
            skel[tuple(line_pts.T)] = True

    return skel


def split_crossings_and_clean(
    binary_mask: np.ndarray,
    k_dir: int = 8,
    theta_straight: float = 150.0,
    L_branch_prune: int = 10,
    remove_cross_node_radius: int = 0,
    min_component_length: int = 0,
    do_bridge: bool = False,
    R_bridge: int = 6,
    bridge_cos_min: float = 0.7,
) -> np.ndarray:
    def _cc_count(mask):
        if mask.sum() == 0:
            return 0
        return int(cc_label(mask.astype(np.uint8), connectivity=mask.ndim).max())

    mask = binary_mask > 0
    if mask.ndim == 2:
        skel = skeletonize(mask).astype(bool)
    elif mask.ndim == 3:
        skel = skeletonize_3d(mask).astype(bool)
    else:
        raise ValueError("binary_mask must be 2D or 3D")

    if skel.sum() == 0:
        return np.zeros_like(binary_mask, dtype=np.int32)

    initial_comp = _cc_count(skel)
    deg = _neighbor_count_nd(skel)
    node_id_map, num_nodes = _build_supernodes(skel, deg)

    if num_nodes == 0:
        sk2 = _prune_short_spurs_simple(skel, L_prune=L_branch_prune)
        pruned_comp = _cc_count(sk2)
        if pruned_comp != initial_comp:
            print(f"[crossing] prune components {initial_comp} -> {pruned_comp}")
        if do_bridge and R_bridge > 0:
            bridge_before = _cc_count(sk2)
            sk2 = _bridge_small_gaps(sk2, R_bridge=R_bridge, cos_min=bridge_cos_min)
            bridge_after = _cc_count(sk2)
        return cc_label(sk2.astype(np.uint8), connectivity=skel.ndim).astype(np.int32)

    edges, node_to_edges = _collect_edges(skel, deg, node_id_map, num_nodes, k_dir=k_dir)

    crossing_nodes = set()
    pairing_info = []
    for node_id in range(1, num_nodes + 1):
        eidxs = node_to_edges.get(node_id, [])
        if len(eidxs) < 4:
            continue
        if len(eidxs) > 4:
            eidxs = sorted(eidxs, key=lambda ei: edges[ei].length, reverse=True)[:4]

        incident_dirs = []
        lengths = []
        for ei in eidxs:
            et = edges[ei]
            if et.start_node == node_id:
                incident_dirs.append(et.dir_from_start)
            else:
                incident_dirs.append(-et.dir_from_start)
            lengths.append(int(et.length))

        if _is_crossing_like(incident_dirs, theta_straight=theta_straight):
            crossing_nodes.add(node_id)
            pairs = _find_best_cross_pairs(
                incident_dirs, theta_straight=theta_straight, lengths=lengths,
            )
            if pairs is not None:
                p1, p2 = pairs
                pairing_info.append(
                    (node_id, (eidxs[p1[0]], eidxs[p1[1]]), (eidxs[p2[0]], eidxs[p2[1]]))
                )

    sk_cut = skel.copy()
    if crossing_nodes:
        structure = np.ones((3,) * skel.ndim, dtype=bool)
        for node_id in crossing_nodes:
            node_mask = node_id_map == node_id
            if remove_cross_node_radius and remove_cross_node_radius > 0:
                node_mask = binary_dilation(
                    node_mask, structure=structure, iterations=int(remove_cross_node_radius),
                )
            sk_cut[node_mask] = False

    comp = cc_label(sk_cut.astype(np.uint8), connectivity=skel.ndim).astype(np.int32)
    num_comp = int(comp.max())
    if crossing_nodes:
        after_cut = int(num_comp)
        if after_cut != initial_comp:
            print(f"[crossing] components {initial_comp} -> {after_cut}")
    if num_comp == 0:
        return np.zeros_like(comp, dtype=np.int32)

    cleaned = {}
    comp_sizes = {}
    discarded_short = 0
    discarded_empty = 0
    for cid in range(1, num_comp + 1):
        comp_skel = comp == cid
        comp_len = int(comp_skel.sum())
        if min_component_length > 0 and comp_len < min_component_length:
            discarded_short += 1
            continue
        comp_skel = _prune_short_spurs_simple(comp_skel, L_prune=L_branch_prune)
        if do_bridge and R_bridge > 0:
            bridge_before = int(comp_skel.sum())
            comp_skel = _bridge_small_gaps(
                comp_skel, R_bridge=R_bridge, cos_min=bridge_cos_min, k_dir=min(6, k_dir),
            )
            bridge_after = int(comp_skel.sum())
            if bridge_after != bridge_before:
                print(f"[bridge] component size {bridge_before} -> {bridge_after}")
        if comp_skel.sum() == 0:
            discarded_empty += 1
            continue
        cleaned[cid] = comp_skel
        comp_sizes[cid] = int(comp_skel.sum())

    if discarded_short or discarded_empty:
        print(
            f"[crossing] discarded components: short={discarded_short}, empty_after_prune={discarded_empty}"
        )

    if not cleaned:
        return np.zeros_like(comp, dtype=np.int32)

    if len(cleaned) != num_comp:
        print(f"[crossing] components {num_comp} -> {len(cleaned)} after cleaning")

    if not pairing_info:
        out = np.zeros_like(comp, dtype=np.int32)
        new_id = 0
        for cid in sorted(cleaned.keys()):
            new_id += 1
            out[cleaned[cid]] = new_id
        return out

    parent = {cid: cid for cid in cleaned.keys()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    pair_groups = []
    for node_id, pair1, pair2 in pairing_info:
        pair1_comps = set()
        for ei in pair1:
            cid = _edge_component_id(edges[ei], node_id, comp, sk_cut)
            if cid in cleaned:
                pair1_comps.add(cid)
        pair2_comps = set()
        for ei in pair2:
            cid = _edge_component_id(edges[ei], node_id, comp, sk_cut)
            if cid in cleaned:
                pair2_comps.add(cid)

        if not pair1_comps or not pair2_comps:
            continue

        p1_list = list(pair1_comps)
        for cid in p1_list[1:]:
            union(p1_list[0], cid)
        p2_list = list(pair2_comps)
        for cid in p2_list[1:]:
            union(p2_list[0], cid)

        pair_groups.append((pair1_comps, pair2_comps))

    group_sizes = {}
    for cid, size in comp_sizes.items():
        root = find(cid)
        group_sizes[root] = group_sizes.get(root, 0) + size

    secondary_groups = set()
    for pair1_comps, pair2_comps in pair_groups:
        root1 = find(next(iter(pair1_comps)))
        root2 = find(next(iter(pair2_comps)))
        if root1 == root2:
            continue
        size1 = group_sizes.get(root1, 0)
        size2 = group_sizes.get(root2, 0)
        if size1 < size2:
            secondary_groups.add(root1)
        elif size2 < size1:
            secondary_groups.add(root2)
        else:
            secondary_groups.add(root2)

    all_roots = {find(cid) for cid in cleaned.keys()}
    primary_roots = all_roots - secondary_groups
    if not primary_roots:
        primary_root = max(all_roots, key=lambda r: group_sizes.get(r, 0))
        secondary_groups.discard(primary_root)
        primary_roots = {primary_root}

    primary_components = [cid for cid in cleaned.keys() if find(cid) in primary_roots]
    if not primary_components:
        return np.zeros_like(comp, dtype=np.int32)

    # Start with the longest primary component as the anchor.
    anchor_cid = max(primary_components, key=lambda cid: comp_sizes.get(cid, 0))
    primary_mask = cleaned[anchor_cid].copy()
    anchor_endpoints = _component_endpoints(primary_mask)

    for cid in primary_components:
        if cid == anchor_cid:
            continue
        comp_mask = cleaned[cid]
        comp_endpoints = _component_endpoints(comp_mask)

        keep = True
        if R_bridge is not None and R_bridge > 0 and anchor_endpoints.size and comp_endpoints.size:
            dist, a_pt, b_pt = _closest_endpoint_pair(anchor_endpoints, comp_endpoints)
            if dist <= float(R_bridge):
                line_pts = _draw_line_nd(a_pt, b_pt)
                primary_mask[tuple(line_pts.T)] = True
            else:
                print(
                    f"[bridge] fail to connect components: min endpoint dist={dist:.3f} (> R_bridge={R_bridge})"
                )
                keep = False

        if keep:
            primary_mask |= comp_mask

    out = np.zeros_like(comp, dtype=np.int32)
    out[primary_mask] = 1

    # Secondary groups each become their own instance label.
    secondary_labels = {}
    next_id = 1
    for root in sorted(secondary_groups):
        next_id += 1
        secondary_labels[root] = next_id

    for cid, comp_skel in cleaned.items():
        root = find(cid)
        if root in secondary_groups:
            out[comp_skel] = secondary_labels[root]

    final_comp = int(out.max())
    if final_comp != len(cleaned):
        print(f"[merge] components {len(cleaned)} -> {final_comp} after crossing pairing")

    return out


def _polylines_to_mask(
    polylines, volume_shape=None, pad: int = 1,
):
    pts_list = []
    for poly in polylines:
        if poly is None:
            continue
        poly = np.asarray(poly, dtype=float)
        if poly.size == 0:
            continue
        dense = connect_neurite_points(poly, max_step=0)
        if dense.size == 0:
            continue
        pts_list.append(dense)

    if not pts_list:
        return None, None

    pts = np.concatenate(pts_list, axis=0)
    pts = np.rint(pts).astype(int)
    if pts.ndim != 2:
        return None, None

    if volume_shape is not None:
        vmax = np.asarray(volume_shape, dtype=int) - 1
        vmin = np.zeros_like(vmax)
        pts = np.clip(pts, vmin, vmax)

    if pts.shape[0] == 0:
        return None, None

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    start = np.maximum(mins - pad, 0)
    end = maxs + pad + 1
    if volume_shape is not None:
        end = np.minimum(end, np.asarray(volume_shape, dtype=int))
    shape = end - start
    if np.any(shape <= 0):
        return None, None

    local = pts - start
    valid = np.all((local >= 0) & (local < shape), axis=1)
    local = local[valid]
    if local.size == 0:
        return None, None

    mask = np.zeros(tuple(shape.tolist()), dtype=bool)
    mask[tuple(local.T)] = True
    return mask, start


def _component_points_to_polylines(points, smooth_win=5, rdp_eps=1.5):
    pts = np.asarray(points, dtype=int)
    if pts.size == 0:
        return []
    if pts.shape[1] != 3:
        return [pts.astype(float, copy=False)]
    simplified = skeleton_to_simplified_paths(
        pts,
        connectivity=26,
        keep_largest_cc=True,
        spur_len=0,
        smooth_win=smooth_win,
        rdp_eps=rdp_eps,
    )
    polylines = [b["polyline"] for b in simplified["branches"]]
    polylines.extend([l["polyline"] for l in simplified["loops"]])
    if not polylines:
        polylines = [pts.astype(float, copy=False)]
    return polylines


def split_crossings_and_clean_polylines(
    polylines,
    volume_shape,
    k_dir: int = 8,
    theta_straight: float = 150.0,
    L_branch_prune: int = 10,
    remove_cross_node_radius: int = 0,
    min_component_length: int = 0,
    R_bridge: int = 0,
    bridge_cos_min: float = 0.7,
    smooth_win: int = 5,
    rdp_eps: float = 1.5,
):
    mask, start = _polylines_to_mask(polylines, volume_shape=volume_shape, pad=1)
    if mask is None:
        return []

    before_count = 0
    if polylines is not None:
        before_count = len(polylines) if isinstance(polylines, list) else 1

    labeled = split_crossings_and_clean(
        mask,
        k_dir=k_dir,
        theta_straight=theta_straight,
        L_branch_prune=L_branch_prune,
        remove_cross_node_radius=remove_cross_node_radius,
        min_component_length=min_component_length,
        do_bridge=(R_bridge is not None and R_bridge > 0),
        R_bridge=R_bridge,
        bridge_cos_min=bridge_cos_min,
    )

    num_comp = int(labeled.max())
    if before_count and num_comp != before_count:
        print(f"[crossing] components {before_count} -> {num_comp} after split_crossings_and_clean")
    if num_comp == 0:
        return []

    components = []
    for cid in range(1, num_comp + 1):
        comp_local = np.argwhere(labeled == cid)
        if comp_local.size == 0:
            continue
        comp_global = comp_local + start
        comp_polys = _component_points_to_polylines(
            comp_global, smooth_win=smooth_win, rdp_eps=rdp_eps,
        )
        components.append(
            {"polylines": comp_polys, "points": comp_global,}
        )
    return components


def lcss_matches_3d(A, B, eps, delta=None, band=None):
    """
    LCSS length and matched index pairs between ordered 3D sequences A and B.
    Returns: (length, matches) where matches is a list of (i, j) index pairs.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    N, M = A.shape[0], B.shape[0]
    if N == 0 or M == 0:
        return 0, []

    if band is None:
        band = delta

    dp = np.zeros((N + 1, M + 1), dtype=np.int32)
    eps2 = float(eps) * float(eps)

    for i in range(1, N + 1):
        if band is None:
            j_start, j_end = 1, M
        else:
            j_start = max(1, i - band)
            j_end = min(M, i + band)
            if j_start > j_end:
                continue
        ai = A[i - 1]
        for j in range(j_start, j_end + 1):
            bj = B[j - 1]
            d2 = np.dot(ai - bj, ai - bj)
            if d2 <= eps2:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                left = dp[i, j - 1]
                up = dp[i - 1, j]
                dp[i, j] = left if left >= up else up

    matches = []
    i, j = N, M
    while i > 0 and j > 0:
        if band is None or abs(i - j) <= band:
            ai = A[i - 1]
            bj = B[j - 1]
            d2 = np.dot(ai - bj, ai - bj)
            if d2 <= eps2 and dp[i, j] == dp[i - 1, j - 1] + 1:
                matches.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if dp[i - 1, j] >= dp[i, j - 1]:
            i -= 1
        else:
            j -= 1

    matches.reverse()
    return int(dp[N, M]), matches


def _polylines_to_coords(polylines, max_step=0):
    if polylines is None:
        return np.zeros((0, 3), dtype=int)
    coords_list = []
    dim = None
    if isinstance(polylines, list):
        items = polylines
    else:
        items = [polylines]
    for poly in items:
        if poly is None:
            continue
        pts = np.asarray(poly, dtype=float)
        if pts.ndim != 2 or pts.shape[0] == 0:
            continue
        if dim is None:
            dim = pts.shape[1]
        elif pts.shape[1] != dim:
            continue
        pts = connect_neurite_points(pts, max_step=max_step)
        if pts.size == 0:
            continue
        coords_list.append(pts)
    if not coords_list:
        if dim is None:
            dim = 3
        return np.zeros((0, dim), dtype=int)
    coords = np.concatenate(coords_list, axis=0)
    coords = np.rint(coords).astype(int)
    return coords


def _coords_to_local_mask(coords: np.ndarray, margin: int = 1):
    if coords is None or coords.size == 0:
        return None, None
    coords = np.asarray(coords, dtype=int)
    mn = coords.min(axis=0) - int(margin)
    mx = coords.max(axis=0) + int(margin)
    shape = (mx - mn + 1).astype(int)
    if np.any(shape <= 0):
        return None, None
    local = coords - mn
    valid = np.all((local >= 0) & (local < shape), axis=1)
    local = local[valid]
    if local.size == 0:
        return None, None
    mask = np.zeros(tuple(shape.tolist()), dtype=bool)
    mask[tuple(local.T)] = True
    return mask, mn


def _trace_inward_path_nd(
    skel: np.ndarray, deg: np.ndarray, start: Tuple[int, ...], max_len: int = 15
):
    offsets = _get_neighbor_offsets(skel.ndim)
    path = [start]
    prev = None
    cur = start
    for _ in range(max_len - 1):
        nbrs = []
        for nbr in _iter_neighbors(cur, skel.shape, offsets):
            if not skel[nbr]:
                continue
            if prev is not None and nbr == prev:
                continue
            nbrs.append(nbr)
        if not nbrs:
            break
        nxt = nbrs[0]
        prev = cur
        cur = nxt
        path.append(cur)
        if deg[cur] >= 3:
            break
    return path


def _estimate_outward_tangent_nd(
    skel: np.ndarray, deg: np.ndarray, ep: Tuple[int, ...], k: int = 8
):
    if k < 2:
        k = 2
    path = _trace_inward_path_nd(skel, deg, ep, max_len=k)
    if len(path) < 2:
        return np.zeros((skel.ndim,), dtype=np.float32)
    kk = min(k - 1, len(path) - 1)
    p0 = np.asarray(path[0], dtype=np.float32)
    pk = np.asarray(path[kk], dtype=np.float32)
    inward = pk - p0
    outward = -inward
    n = np.linalg.norm(outward) + 1e-8
    return (outward / n).astype(np.float32)


def _segment_endpoints_and_tangents(coords: np.ndarray, k_tangent: int = 8):
    skel, offset = _coords_to_local_mask(coords, margin=1)
    if skel is None or skel.sum() == 0:
        return []
    deg = _neighbor_count_nd(skel)
    eps = np.argwhere(skel & (deg == 1))
    if eps.size == 0:
        return []
    out = []
    offset = offset.astype(np.float32)
    for ep in eps:
        ep_t = tuple(int(v) for v in ep)
        t = _estimate_outward_tangent_nd(skel, deg, ep_t, k=k_tangent)
        p = ep.astype(np.float32) + offset
        out.append((p, t))
    return out


def _find_junction_components(union_mask: np.ndarray):
    if union_mask.ndim == 2:
        skel_u = skeletonize(union_mask).astype(bool)
    elif union_mask.ndim == 3:
        skel_u = skeletonize_3d(union_mask).astype(bool)
    else:
        raise ValueError("union_mask must be 2D or 3D")
    if skel_u.sum() == 0:
        return []
    deg_u = _neighbor_count_nd(skel_u)
    jmask = skel_u & (deg_u >= 3)
    jcc = cc_label(jmask.astype(np.uint8), connectivity=skel_u.ndim)
    comps = []
    for jid in range(1, int(jcc.max()) + 1):
        pts = np.argwhere(jcc == jid).astype(np.float32)
        if pts.size:
            comps.append(pts)
    return comps


def _render_merge_pair_debug(
    coords_a: np.ndarray, coords_b: np.ndarray, coords_merged: np.ndarray, volume_shape, out_path,
):
    vol_shape = np.asarray(volume_shape, dtype=int)
    if vol_shape.size == 3:
        h, w = int(vol_shape[1]), int(vol_shape[2])

        def _to_yx(coords):
            if coords.size == 0:
                return np.zeros((0, 2), dtype=int)
            y = coords[:, 1]
            x = coords[:, 2]
            return np.stack([y, x], axis=1)

    elif vol_shape.size == 2:
        h, w = int(vol_shape[0]), int(vol_shape[1])

        def _to_yx(coords):
            if coords.size == 0:
                return np.zeros((0, 2), dtype=int)
            return coords[:, :2]

    else:
        return

    def _mask_from_coords(coords):
        if coords is None or coords.size == 0:
            return np.zeros((h, w), dtype=bool)
        coords = np.asarray(coords, dtype=int)
        yx = _to_yx(coords)
        if yx.size == 0:
            return np.zeros((h, w), dtype=bool)
        y = np.clip(yx[:, 0], 0, h - 1)
        x = np.clip(yx[:, 1], 0, w - 1)
        mask = np.zeros((h, w), dtype=bool)
        mask[y, x] = True
        return mask

    mask_a = _mask_from_coords(coords_a)
    mask_b = _mask_from_coords(coords_b)
    mask_m = _mask_from_coords(coords_merged)

    left = np.zeros((h, w, 3), dtype=np.uint8)
    right = np.zeros((h, w, 3), dtype=np.uint8)

    left[mask_a] = (0, 0, 255)  # blue
    left[mask_b] = (255, 0, 0)  # red
    overlap = mask_a & mask_b
    if overlap.any():
        left[overlap] = (255, 0, 255)  # magenta

    right[mask_m] = (255, 255, 255)  # white

    out = np.zeros((h, w * 2, 3), dtype=np.uint8)
    out[:, :w] = left
    out[:, w:] = right

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path.as_posix(), out)


def merge_instances_joint(
    skeleton_paths: Dict[int, List[np.ndarray]],
    inst_neur: Dict[int, int],
    volume_shape,
    max_dist: float = 2.0,
    min_lcss: int = 5,
    endpoint_min_overlap: int = 5,
    endpoint_radius: Optional[float] = None,
    max_index_jump: Optional[int] = None,
    max_step: int = 16,
    merge_overlap: bool = True,
    merge_tip: bool = True,
    cover_ratio: float = 0.9,
    min_short_points: int = 8,
    long_min_points: int = 80,
    short_to_long_max_ratio: float = 0.6,
    k_prefix: int = 15,
    min_prefix_overlap: int = 5,
    densify_step: int = 1,
    take_polylines: int = 2,
    candidate_radius: Optional[float] = None,
    use_bbox_gate: bool = True,
    Rj: float = 6.0,
    Rd: float = 6.0,
    R_global: float = 4.0,
    theta_deg: float = 45.0,
    k_tangent: int = 8,
    greedy_one_pair_per_junction: bool = True,
    do_lcss: bool = True,
    debug_dir: Optional[pathlib.Path] = None,
):
    if not skeleton_paths or len(skeleton_paths) < 2:
        return dict(skeleton_paths), dict(inst_neur)

    if endpoint_radius is None:
        endpoint_radius = float(max_dist) * 2.0
    if candidate_radius is None:
        candidate_radius = float(max_dist) * 4.0
    if max_index_jump is None:
        max_index_jump = max(3, int(min_lcss // 2))

    vol_shape = np.asarray(volume_shape, dtype=int)
    if vol_shape.ndim != 1 or vol_shape.size not in (2, 3):
        raise ValueError(f"volume_shape must be 2D or 3D, got {volume_shape}")

    seg_ids = sorted([int(k) for k in skeleton_paths.keys()])
    parent = {sid: sid for sid in seg_ids}
    merge_debug_dir = pathlib.Path(debug_dir) if debug_dir is not None else None
    merge_debug_idx = 0

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _debug_pair(a, b, tag):
        nonlocal merge_debug_idx
        if merge_debug_dir is None:
            return
        coords_a = coords_cache.get(a, np.zeros((0, 3), dtype=int))
        coords_b = coords_cache.get(b, np.zeros((0, 3), dtype=int))
        if coords_a.size == 0 and coords_b.size == 0:
            return
        coords_m = (
            coords_a
            if coords_b.size == 0
            else coords_b
            if coords_a.size == 0
            else np.concatenate([coords_a, coords_b], axis=0)
        )
        out_path = merge_debug_dir / f"merge_{merge_debug_idx:05d}_{a}_{b}_{tag}.png"
        merge_debug_idx += 1
        _render_merge_pair_debug(coords_a, coords_b, coords_m, vol_shape, out_path)

    def union(a, b, tag="merge"):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            _debug_pair(a, b, tag)
            parent[rb] = ra

    def _as_list_polys(polys):
        if polys is None:
            return []
        return polys if isinstance(polys, list) else [polys]

    def _clip_to_volume(coords):
        if coords is None or coords.size == 0:
            return np.zeros((0, vol_shape.size), dtype=coords.dtype if coords is not None else int)
        valid = np.ones(coords.shape[0], dtype=bool)
        for axis, dim in enumerate(vol_shape):
            valid &= (coords[:, axis] >= 0) & (coords[:, axis] < int(dim))
        return coords[valid]

    def _densify_poly(poly):
        pts = np.asarray(poly, dtype=float)
        if pts.ndim != 2 or pts.shape[0] == 0:
            dim = pts.shape[1] if pts.ndim == 2 else int(vol_shape.size)
            return np.zeros((0, dim), dtype=np.float32)
        dense = connect_neurite_points(pts, max_step=int(densify_step))
        return dense.astype(np.float32, copy=False)

    def _to_dense_coords(polys):
        coords = _polylines_to_coords(polys, max_step=int(densify_step))
        return coords.astype(np.float32, copy=False)

    def _extract_prefix_blocks(polys):
        blocks = []
        densified = []
        for poly in polys:
            if poly is None:
                continue
            d = _densify_poly(poly)
            if d.shape[0] > 0:
                densified.append(d)
        if not densified:
            return blocks
        densified.sort(key=lambda x: x.shape[0], reverse=True)
        densified = densified[: int(take_polylines)]
        kp = int(k_prefix)
        for d in densified:
            n = d.shape[0]
            if n <= 0:
                continue
            k = min(kp, n)
            blocks.append(d[:k])
            blocks.append(d[-k:])
        return blocks

    coords_cache = {}
    endpoints = {}
    union_mask = np.zeros(tuple(vol_shape.tolist()), dtype=bool)
    info = {}
    endpoint_points = []
    endpoint_owner = []
    do_overlap = bool(merge_overlap)
    do_tip = bool(merge_tip)
    do_new_merges = do_overlap or do_tip
    for sid in seg_ids:
        polylines = skeleton_paths.get(sid, [])
        coords = _polylines_to_coords(polylines, max_step=max_step)
        coords = _clip_to_volume(coords)
        coords_cache[sid] = coords
        if coords.size:
            union_mask[tuple(coords.T)] = True

        if not do_new_merges:
            continue

        dense_coords = _to_dense_coords(polylines)
        dense_coords = _clip_to_volume(dense_coords)
        if dense_coords.size == 0:
            info[sid] = None
            continue
        npts = int(dense_coords.shape[0])
        bb_min = dense_coords.min(axis=0)
        bb_max = dense_coords.max(axis=0)
        prefix_blocks = _extract_prefix_blocks(_as_list_polys(polylines)) if do_tip else []
        ep_list = []
        for blk in prefix_blocks:
            if blk is None or blk.size == 0:
                continue
            ep_list.append(blk[0])
            ep_list.append(blk[-1])
        if not ep_list:
            ep_list = [dense_coords[0], dense_coords[-1]]
        info[sid] = {
            "coords": dense_coords,
            "npts": npts,
            "bb_min": bb_min,
            "bb_max": bb_max,
            "prefix_blocks": prefix_blocks,
            "endpoints": ep_list,
        }
        for ep in ep_list[:4]:
            endpoint_points.append(ep.astype(np.float32, copy=False))
            endpoint_owner.append(sid)

    # 0) Short-long containment + long-long endpoint-prefix overlap merging.
    if do_new_merges and len(endpoint_points) >= 2:
        endpoint_points = np.stack(endpoint_points, axis=0)
        ep_tree = cKDTree(endpoint_points)

        candidates = set()
        for i, p in enumerate(endpoint_points):
            sid_i = endpoint_owner[i]
            nbrs = ep_tree.query_ball_point(p, r=float(candidate_radius))
            for j in nbrs:
                if j <= i:
                    continue
                sid_j = endpoint_owner[j]
                if sid_i == sid_j:
                    continue
                a, b = (sid_i, sid_j) if sid_i < sid_j else (sid_j, sid_i)
                candidates.add((a, b))

        if use_bbox_gate and candidates:
            gated = set()
            margin = float(candidate_radius)
            for a, b in candidates:
                ia, ib = info.get(a), info.get(b)
                if ia is None or ib is None:
                    continue
                if np.all(ia["bb_min"] <= ib["bb_max"] + margin) and np.all(
                    ib["bb_min"] <= ia["bb_max"] + margin
                ):
                    gated.add((a, b))
            candidates = gated

        coord_tree_cache = {}
        prefix_tree_cache = {}

        def _get_coord_tree(sid):
            if sid not in coord_tree_cache:
                coord_tree_cache[sid] = cKDTree(info[sid]["coords"])
            return coord_tree_cache[sid]

        def _get_prefix_trees(sid):
            if sid in prefix_tree_cache:
                return prefix_tree_cache[sid]
            blocks = info[sid]["prefix_blocks"]
            out = []
            for blk in blocks:
                if blk is None or blk.size == 0:
                    continue
                out.append((blk, cKDTree(blk)))
            prefix_tree_cache[sid] = out
            return out

        r = float(max_dist)
        for a, b in candidates:
            if find(a) == find(b):
                continue
            ia, ib = info.get(a), info.get(b)
            if ia is None or ib is None:
                continue
            na, nb = ia["npts"], ib["npts"]

            if na <= nb:
                short_id, long_id = a, b
                ns, nl = na, nb
            else:
                short_id, long_id = b, a
                ns, nl = nb, na

            if do_overlap and (
                ns >= int(min_short_points)
                and nl >= int(long_min_points)
                and (
                    short_to_long_max_ratio is None
                    or float(ns) <= float(nl) * float(short_to_long_max_ratio)
                )
            ):
                tree_long = _get_coord_tree(long_id)
                dists, _ = tree_long.query(info[short_id]["coords"], k=1)
                covered = float(np.mean(dists <= r)) if dists.size else 0.0
                if covered >= float(cover_ratio):
                    union(long_id, short_id, tag="contained")
                    continue

            if do_tip and na >= int(long_min_points) and nb >= int(long_min_points):
                trees_a = _get_prefix_trees(a)
                trees_b = _get_prefix_trees(b)
                if trees_a and trees_b:
                    merged = False
                    for blk_a, _ta in trees_a:
                        for _blk_b, tb in trees_b:
                            dists, _ = tb.query(blk_a, k=1)
                            overlap = int(np.sum(dists <= r))
                            if overlap >= int(min_prefix_overlap):
                                union(a, b, tag="endpoint_prefix")
                                merged = True
                                break
                        if merged:
                            break

    for sid in seg_ids:
        coords = coords_cache.get(sid, np.zeros((0, vol_shape.size), dtype=int))
        endpoints[sid] = _segment_endpoints_and_tangents(coords, k_tangent=k_tangent)

    cos_theta = float(np.cos(np.deg2rad(theta_deg)))

    def _continuation_ok(p1, t1, p2, t2):
        v = p2 - p1
        d = float(np.linalg.norm(v))
        if d < 1e-6:
            return False, d, 0.0
        u = v / d
        a1 = float(np.dot(t1, u))
        a2 = float(np.dot(t2, -u))
        ok = (a1 >= cos_theta) and (a2 >= cos_theta)
        align = 0.5 * (a1 + a2)
        return ok, d, align

    def _score_pair(d, align, w_d=0.2, w_align=1.0):
        return float(w_align * align - w_d * d)

    # 1) Junction-based merging (endpoint evidence).
    if union_mask.any():
        junctions = _find_junction_components(union_mask)
    else:
        junctions = []

    pre_junction_count = len(seg_ids)
    for Jpts in junctions:
        if Jpts.size == 0:
            continue
        tree_j = cKDTree(Jpts)
        cand = []
        for sid in seg_ids:
            for (p, t) in endpoints.get(sid, []):
                if tree_j.query(p, k=1)[0] <= float(Rj):
                    cand.append((sid, p, t))
        if len(cand) < 2:
            continue

        if greedy_one_pair_per_junction:
            best = None
            best_s = -1e9
            for i in range(len(cand)):
                sid1, p1, t1 = cand[i]
                for j in range(i + 1, len(cand)):
                    sid2, p2, t2 = cand[j]
                    if find(sid1) == find(sid2):
                        continue
                    ok, d, align = _continuation_ok(p1, t1, p2, t2)
                    if not ok or d > float(Rd):
                        continue
                    s = _score_pair(d, align)
                    if s > best_s:
                        best_s = s
                        best = (sid1, sid2)
            if best is not None:
                union(best[0], best[1], tag="junction")
        else:
            pairs = []
            for i in range(len(cand)):
                sid1, p1, t1 = cand[i]
                for j in range(i + 1, len(cand)):
                    sid2, p2, t2 = cand[j]
                    if find(sid1) == find(sid2):
                        continue
                    ok, d, align = _continuation_ok(p1, t1, p2, t2)
                    if ok and d <= float(Rd):
                        pairs.append((_score_pair(d, align), i, j, sid1, sid2))
            pairs.sort(reverse=True, key=lambda x: x[0])
            used = set()
            for _s, i, j, sid1, sid2 in pairs:
                if i in used or j in used:
                    continue
                if find(sid1) == find(sid2):
                    continue
                union(sid1, sid2, tag="junction")
                used.add(i)
                used.add(j)

    # 2) Global endpoint-close merging (direction gated).
    all_eps = []
    for sid in seg_ids:
        for (p, t) in endpoints.get(sid, []):
            all_eps.append((sid, p, t))
    if len(all_eps) >= 2:
        ep_coords = np.stack([x[1] for x in all_eps], axis=0)
        tree = cKDTree(ep_coords)
        for i in range(len(all_eps)):
            sid1, p1, t1 = all_eps[i]
            nbrs = tree.query_ball_point(p1, r=float(R_global))
            for j in nbrs:
                if j <= i:
                    continue
                sid2, p2, t2 = all_eps[j]
                if find(sid1) == find(sid2):
                    continue
                ok, _d, _align = _continuation_ok(p1, t1, p2, t2)
                if ok:
                    union(sid1, sid2, tag="endpoint")

    post_endpoint_count = len({find(sid) for sid in seg_ids})
    if post_endpoint_count != pre_junction_count:
        print(
            f"[merge] instances {pre_junction_count} -> {post_endpoint_count} after junction/endpoint merge"
        )

    if do_lcss:
        # ---- Precompute endpoint neighborhoods for overlap test ----
        neighborhoods = {}
        endpoint_pts = []
        endpoint_meta = []
        for sid in seg_ids:
            coords = coords_cache.get(sid, np.zeros((0, 3), dtype=int))
            if coords.size == 0:
                neighborhoods[sid] = []
                continue
            eps = endpoints.get(sid, [])
            if not eps:
                neighborhoods[sid] = []
                continue
            tree = cKDTree(coords.astype(float))
            neigh_list = []
            for e_i, (p, _t) in enumerate(eps):
                idxs = tree.query_ball_point(p, r=float(endpoint_radius))
                if idxs:
                    neigh = coords[np.array(idxs, dtype=int)]
                else:
                    neigh = np.zeros((0, coords.shape[1]), dtype=int)
                neigh_list.append(neigh)
                endpoint_pts.append(p.astype(float))
                endpoint_meta.append((sid, e_i))
            neighborhoods[sid] = neigh_list

        if len(endpoint_pts) >= 2:
            endpoint_pts = np.stack(endpoint_pts, axis=0)
            ep_tree = cKDTree(endpoint_pts)

            candidates = set()
            for i, p in enumerate(endpoint_pts):
                nbrs = ep_tree.query_ball_point(p, r=float(endpoint_radius))
                sid_i, _ = endpoint_meta[i]
                for j in nbrs:
                    if j <= i:
                        continue
                    sid_j, _ = endpoint_meta[j]
                    if sid_i == sid_j:
                        continue
                    a, b = (sid_i, sid_j) if sid_i < sid_j else (sid_j, sid_i)
                    candidates.add((a, b))

            def endpoint_overlap_ok(a, b):
                neigh_a = neighborhoods.get(a, [])
                neigh_b = neighborhoods.get(b, [])
                if not neigh_a or not neigh_b:
                    return False
                r = float(max_dist)
                K = int(endpoint_min_overlap)
                for a_set in neigh_a:
                    if a_set.size == 0:
                        continue
                    tree_a = cKDTree(a_set.astype(float))
                    for b_set in neigh_b:
                        if b_set.size == 0:
                            continue
                        dists, _ = tree_a.query(b_set.astype(float), k=1)
                        if np.count_nonzero(dists <= r) >= K:
                            return True
                return False

            def lcss_ok(a, b):
                eps = float(max_dist)
                min_match = max(3, int(min_lcss))
                jump = int(max_index_jump)

                polys_a = (
                    skeleton_paths[a]
                    if isinstance(skeleton_paths[a], list)
                    else [skeleton_paths[a]]
                )
                polys_b = (
                    skeleton_paths[b]
                    if isinstance(skeleton_paths[b], list)
                    else [skeleton_paths[b]]
                )

                for pa in polys_a:
                    pts_a = np.asarray(pa, dtype=float)
                    if pts_a.ndim != 2 or pts_a.shape[0] < 2:
                        continue
                    for pb in polys_b:
                        pts_b = np.asarray(pb, dtype=float)
                        if pts_b.ndim != 2 or pts_b.shape[0] < 2:
                            continue
                        lcss_len, matches = lcss_matches_3d(pts_a, pts_b, eps)
                        if lcss_len < min_match or not matches:
                            continue
                        idx_a = [i for i, _ in matches]
                        idx_b = [j for _, j in matches]
                        if len(idx_a) > 1 and np.max(np.diff(idx_a)) > jump:
                            continue
                        if len(idx_b) > 1 and np.max(np.diff(idx_b)) > jump:
                            continue
                        return True
                return False

            for a, b in candidates:
                if find(a) == find(b):
                    continue
                if endpoint_overlap_ok(a, b):
                    union(a, b, tag="endpoint_overlap")
                elif lcss_ok(a, b):
                    union(a, b, tag="lcss")

            post_lcss_count = len({find(sid) for sid in seg_ids})
            if post_lcss_count != post_endpoint_count:
                print(
                    f"[merge] instances {post_endpoint_count} -> {post_lcss_count} after endpoint-overlap/LCSS"
                )

    merged_paths = defaultdict(list)
    merged_neur = defaultdict(list)
    for sid, polylines in skeleton_paths.items():
        root = find(int(sid))
        merged_paths[root].extend(polylines if isinstance(polylines, list) else [polylines])
        merged_neur[root].append(inst_neur.get(sid, -1))

    out_paths = dict(merged_paths)
    out_neur = {}
    for root, neur_ids in merged_neur.items():
        valid = {n for n in neur_ids if n != -1}
        out_neur[root] = valid.pop() if len(valid) == 1 else -1

    return out_paths, out_neur


def skeletonize_pred(
    preds,
    volume_shape,
    output_patch_size=1,
    min_overlap_length=5,
    max_dist=2.0,
    max_step=16,
    merge=True,
    merge_overlap=False,
    merge_tip=False,
    break_branch=False,
    return_debug_info=False,
    crossing_k_dir=8,
    crossing_theta_straight=150.0,
    crossing_L_branch_prune=10,
    crossing_remove_cross_node_radius=0,
    crossing_min_component_length=0,
    R_bridge=0,
    crossing_bridge_cos_min=0.7,
    post_break_min_length=0,
    trace_instance=True,
    min_length=1,
    merge_all_instance=False,
):
    pred_centers, pred_inst, pred_neur = preds["centers"], preds["inst_ids"], preds["neuron_ids"]
    if torch.is_tensor(pred_centers):
        pred_centers = pred_centers.detach().cpu().numpy()
    else:
        pred_centers = np.asarray(pred_centers)
    if torch.is_tensor(pred_inst):
        pred_inst = pred_inst.detach().cpu().numpy()
    else:
        pred_inst = np.asarray(pred_inst)
    if torch.is_tensor(pred_neur):
        pred_neur = pred_neur.detach().cpu().numpy()
    else:
        pred_neur = np.asarray(pred_neur)

    empty_pred = {
        "centers": np.zeros((0, 3), dtype=float),
        "inst_ids": np.zeros((0,), dtype=int),
        "neuron_ids": np.zeros((0,), dtype=int),
    }

    vol_shape = np.asarray(volume_shape, dtype=int)
    if merge_all_instance:
        coords = (
            np.rint(pred_centers).astype(int) if pred_centers.size else np.zeros((0, 3), dtype=int)
        )
        if coords.size > 0:
            coords = np.clip(coords, [0, 0, 0], vol_shape - 1)
            coords = np.unique(coords, axis=0)

        volume = np.zeros(tuple(vol_shape.tolist()), dtype=bool)
        if coords.shape[0] > 0:
            volume[coords[:, 0], coords[:, 1], coords[:, 2]] = True

        volume = binary_dilation(volume, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
        skeleton = skeletonize_3d(volume)
        skel_pts = np.argwhere(skeleton)
        semantic_points = {
            "centers": skel_pts.astype(float, copy=False),
            "inst_ids": np.zeros((skel_pts.shape[0],), dtype=int),
            "neuron_ids": np.full((skel_pts.shape[0],), -1, dtype=int),
        }

        if return_debug_info:
            return None, semantic_points, None
        return None, semantic_points

    preds_cleaned = None

    inst_ids = np.unique(pred_inst)
    skeleton_paths = {}
    raw_skeleton_pts = {}
    inst_neur = {}
    simplified_points = {}
    simplified_neur = {}
    raw_centers_list = []
    raw_inst_list = []
    raw_neur_list = []
    all_dilated_coords = []
    skipped_instances = 0
    debug_dir = pathlib.Path("eval") / "debug_skeletons"
    debug_dir.mkdir(parents=True, exist_ok=True)
    next_branch_inst_id = int(inst_ids.max()) + 1 if inst_ids.size else 0
    for inst_id in inst_ids:
        mask = pred_inst == inst_id
        centers = pred_centers[mask]
        if centers.size == 0:
            continue
        neur_id = int(pred_neur[mask][0]) if pred_neur.size == pred_inst.size else -1

        coords = np.rint(centers).astype(int)
        coords = np.clip(coords, [0, 0, 0], vol_shape - 1)
        coords = np.unique(coords, axis=0)
        if trace_instance and coords.shape[0] < min_length:
            skipped_instances += 1
            continue

        else:
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            start = np.maximum(mins - 1, 0)
            end = np.minimum(maxs + 2, vol_shape)
            sub_shape = end - start

            volume = np.zeros(tuple(sub_shape.tolist()), dtype=bool)
            sub_coords = coords - start
            volume[sub_coords[:, 0], sub_coords[:, 1], sub_coords[:, 2]] = True

            volume = binary_dilation(volume, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
            skeleton = skeletonize_3d(volume)
            skel_pts = np.argwhere(skeleton)
            if trace_instance and skel_pts.size < min_length:
                skipped_instances += 1
                continue
            else:
                skel_pts = skel_pts + start
            all_dilated_coords.append(np.argwhere(volume) + start)

        if not trace_instance:
            raw_pts = skel_pts.astype(float, copy=False)
            if raw_pts.size > 0 and int(output_patch_size) > 1:
                connect_step = max(1, int(output_patch_size))
                simplified_raw = skeleton_to_simplified_paths(
                    skel_pts,
                    connectivity=26,
                    keep_largest_cc=False,
                    spur_len=0,
                    smooth_win=1,
                    rdp_eps=0.0,
                    max_step=connect_step,
                )
                raw_polylines = [b["polyline"] for b in simplified_raw["branches"]]
                raw_polylines.extend([l["polyline"] for l in simplified_raw["loops"]])
                if not raw_polylines:
                    raw_polylines = [raw_pts]

                dense_raw = []
                for poly in raw_polylines:
                    if poly is None:
                        continue
                    poly = np.asarray(poly, dtype=float)
                    if poly.size == 0:
                        continue
                    dense = connect_neurite_points(poly, max_step=connect_step)
                    if dense.size == 0:
                        continue
                    dense_raw.append(dense)
                if dense_raw:
                    raw_pts = np.concatenate(dense_raw, axis=0)

            if raw_pts.size > 0:
                raw_centers_list.append(raw_pts)
                raw_inst_list.append(np.full(raw_pts.shape[0], inst_id, dtype=int))
                raw_neur_list.append(np.full(raw_pts.shape[0], neur_id, dtype=int))
            continue

        simplified = skeleton_to_simplified_paths(
            skel_pts,
            connectivity=26,
            keep_largest_cc=True,
            spur_len=5,
            smooth_win=5,
            rdp_eps=1.5,
            max_step=max_step,
        )
        polylines = [b["polyline"] for b in simplified["branches"]]
        polylines.extend([l["polyline"] for l in simplified["loops"]])
        if not polylines:
            polylines = [skel_pts.astype(float, copy=False)]
        dense_simplified = []
        for poly in polylines:
            if poly is None:
                continue
            poly = np.asarray(poly, dtype=float)
            if poly.size == 0:
                continue
            dense = connect_neurite_points(poly, max_step=max_step)
            if dense.size == 0:
                continue
            dense_simplified.append(dense)
        if dense_simplified:
            simplified_points[inst_id] = np.concatenate(dense_simplified, axis=0)
        else:
            simplified_points[inst_id] = np.zeros((0, 3), dtype=float)
        simplified_neur[inst_id] = neur_id
        left_img = np.zeros((512, 512), dtype=np.uint8)
        if skel_pts.size > 0:
            xy = skel_pts[:, 1:3].astype(np.int64, copy=False)
            in_bounds = (xy[:, 0] >= 0) & (xy[:, 0] < 512) & (xy[:, 1] >= 0) & (xy[:, 1] < 512)
            xy = xy[in_bounds]
            if xy.size:
                left_img[xy[:, 0], xy[:, 1]] = 255

        simplified_pts = None
        if polylines:
            if len(polylines) == 1:
                simplified_pts = np.asarray(polylines[0])
            else:
                simplified_pts = np.concatenate(polylines, axis=0)
        right_img = np.zeros((512, 512), dtype=np.uint8)
        if simplified_pts is not None and simplified_pts.size > 0:
            xy = np.rint(simplified_pts[:, 1:3]).astype(np.int64, copy=False)
            in_bounds = (xy[:, 0] >= 0) & (xy[:, 0] < 512) & (xy[:, 1] >= 0) & (xy[:, 1] < 512)
            xy = xy[in_bounds]
            if xy.size:
                right_img[xy[:, 0], xy[:, 1]] = 255

        out_img = np.zeros((512, 1024), dtype=np.uint8)
        out_img[:, :512] = left_img
        out_img[:, 512:] = right_img
        imageio.imwrite((debug_dir / f"{inst_id}.png").as_posix(), out_img)

        components = split_crossings_and_clean_polylines(
            polylines,
            volume_shape=vol_shape,
            k_dir=crossing_k_dir,
            theta_straight=crossing_theta_straight,
            L_branch_prune=crossing_L_branch_prune,
            remove_cross_node_radius=crossing_remove_cross_node_radius,
            min_component_length=crossing_min_component_length,
            R_bridge=R_bridge,
            bridge_cos_min=crossing_bridge_cos_min,
            smooth_win=5,
            rdp_eps=1.5,
        )
        if not components:
            components = [{"polylines": polylines, "points": skel_pts,}]

        first_component = True
        for comp in components:
            comp_polys = comp.get("polylines", [])
            comp_pts = comp.get("points", None)
            if not comp_polys:
                if comp_pts is None or comp_pts.size == 0:
                    continue
                comp_polys = [comp_pts.astype(float, copy=False)]

            if first_component:
                comp_id = inst_id
            else:
                comp_id = next_branch_inst_id
                next_branch_inst_id += 1
            skeleton_paths[comp_id] = comp_polys
            raw_skeleton_pts[comp_id] = comp_pts if comp_pts is not None else skel_pts
            inst_neur[comp_id] = neur_id
            first_component = False

    if not trace_instance:
        if raw_centers_list:
            raw_skeleton = {
                "centers": np.concatenate(raw_centers_list, axis=0),
                "inst_ids": np.concatenate(raw_inst_list, axis=0),
                "neuron_ids": np.concatenate(raw_neur_list, axis=0),
            }
        else:
            raw_skeleton = {
                "centers": np.zeros((0, 3), dtype=float),
                "inst_ids": np.zeros((0,), dtype=int),
                "neuron_ids": np.zeros((0,), dtype=int),
            }
        if return_debug_info:
            return None, raw_skeleton, None
        return None, raw_skeleton
    print(
        f"Total predicted instances: {len(inst_ids)}, extracted skeleton paths for "
        f"{len(skeleton_paths)} instances (after splitting). skipped {skipped_instances} "
        "instances with too few skeleton points."
    )

    print(f"Extracted skeleton paths for {len(skeleton_paths)} instances.")
    if not skeleton_paths:
        print("No skeleton paths extracted from predictions.")
        simple_centers_list = []
        simple_inst_list = []
        simple_neur_list = []
        for inst_id, pts in simplified_points.items():
            if pts is None:
                continue
            pts = np.asarray(pts, dtype=float)
            if pts.size == 0:
                continue
            simple_centers_list.append(pts)
            simple_inst_list.append(np.full(pts.shape[0], inst_id, dtype=int))
            neur_id = simplified_neur.get(inst_id, -1)
            simple_neur_list.append(np.full(pts.shape[0], neur_id, dtype=int))
        if simple_centers_list:
            simplified_pred = {
                "centers": np.concatenate(simple_centers_list, axis=0),
                "inst_ids": np.concatenate(simple_inst_list, axis=0),
                "neuron_ids": np.concatenate(simple_neur_list, axis=0),
            }
        else:
            simplified_pred = {
                "centers": np.zeros((0, 3), dtype=float),
                "inst_ids": np.zeros((0,), dtype=int),
                "neuron_ids": np.zeros((0,), dtype=int),
            }
        empty_pred = {
            "centers": np.zeros((0, 3), dtype=float),
            "inst_ids": np.zeros((0,), dtype=int),
            "neuron_ids": np.zeros((0,), dtype=int),
        }
        if return_debug_info:
            return empty_pred, simplified_pred, None
        return empty_pred, simplified_pred

    simple_centers_list = []
    simple_inst_list = []
    simple_neur_list = []
    for inst_id, pts in simplified_points.items():
        if pts is None:
            continue
        pts = np.asarray(pts, dtype=float)
        if pts.size == 0:
            continue
        simple_centers_list.append(pts)
        simple_inst_list.append(np.full(pts.shape[0], inst_id, dtype=int))
        neur_id = simplified_neur.get(inst_id, -1)
        simple_neur_list.append(np.full(pts.shape[0], neur_id, dtype=int))
    if simple_centers_list:
        simplified_pred = {
            "centers": np.concatenate(simple_centers_list, axis=0),
            "inst_ids": np.concatenate(simple_inst_list, axis=0),
            "neuron_ids": np.concatenate(simple_neur_list, axis=0),
        }
    else:
        simplified_pred = {
            "centers": np.zeros((0, 3), dtype=float),
            "inst_ids": np.zeros((0,), dtype=int),
            "neuron_ids": np.zeros((0,), dtype=int),
        }

    pre_merge_count = len(skeleton_paths)
    out_points, out_neur = merge_instances_joint(
        skeleton_paths,
        inst_neur,
        volume_shape=vol_shape,
        max_dist=max_dist,
        min_lcss=min_overlap_length,
        max_step=max_step,
        do_lcss=merge,
        merge_overlap=merge_overlap,
        merge_tip=merge_tip,
        debug_dir=debug_dir / "merge_pairs",
    )
    if len(out_points) != pre_merge_count:
        print(f"Merged to {len(out_points)} instances after joint endpoint/LCSS merging.")
    if break_branch:
        pre_branch_count = len(out_points)
        branched_points = {}
        branched_neur = {}
        next_id = int(max(out_points.keys())) + 1 if out_points else 0
        for inst_id, pts in out_points.items():
            polylines = pts if isinstance(pts, list) else [pts]
            first_poly = True
            for poly in polylines:
                if poly is None:
                    continue
                poly = np.asarray(poly, dtype=float)
                if poly.size == 0:
                    continue
                if first_poly:
                    branch_id = inst_id
                    first_poly = False
                else:
                    branch_id = next_id
                    next_id += 1
                branched_points[branch_id] = [poly]
                branched_neur[branch_id] = out_neur.get(inst_id, -1)
        out_points = branched_points
        out_neur = branched_neur
        print(f"[break_branch] instances {pre_branch_count} -> {len(out_points)}")

    if post_break_min_length and post_break_min_length > 0:
        filtered_points = {}
        filtered_neur = {}
        for inst_id, pts in out_points.items():
            if isinstance(pts, list):
                length = 0
                for p in pts:
                    if p is None:
                        continue
                    p = np.asarray(p)
                    if p.size == 0:
                        continue
                    length += int(p.shape[0])
            else:
                length = int(np.asarray(pts).shape[0]) if pts is not None else 0
            if length >= int(post_break_min_length):
                filtered_points[inst_id] = pts
                filtered_neur[inst_id] = out_neur.get(inst_id, -1)
        out_points = filtered_points
        out_neur = filtered_neur
    print(
        f"Filtered to {len(out_points)} instances after post-merge length filtering (min_length={post_break_min_length})."
    )

    centers_list = []
    inst_list = []
    neur_list = []
    for inst_id, pts in out_points.items():
        if isinstance(pts, list):
            pts = (
                np.concatenate([connect_neurite_points(p, max_step=max_step) for p in pts], axis=0)
                if pts
                else np.zeros((0, 3), dtype=float)
            )
        else:
            pts = connect_neurite_points(pts, max_step=max_step)
        centers_list.append(pts)
        inst_list.append(np.full(pts.shape[0], inst_id, dtype=int))
        neur_id = out_neur.get(inst_id, -1)
        neur_list.append(np.full(pts.shape[0], neur_id, dtype=int))
    if centers_list:
        preds_cleaned = {
            "centers": np.concatenate(centers_list, axis=0),
            "inst_ids": np.concatenate(inst_list, axis=0),
            "neuron_ids": np.concatenate(neur_list, axis=0),
        }
    else:
        preds_cleaned = {
            "centers": np.zeros((0, 3), dtype=float),
            "inst_ids": np.zeros((0,), dtype=int),
            "neuron_ids": np.zeros((0,), dtype=int),
        }

    if return_debug_info:
        # Prepare debug info
        # raw_points: all raw skeleton points before merging
        raw_points = []
        for inst_id, skel_pts in raw_skeleton_pts.items():
            raw_points.append(skel_pts.astype(float))

        # matched_points: points from final instances
        matched_points = []
        for inst_id, pts in out_points.items():
            if isinstance(pts, list):
                for p in pts:
                    matched_points.append(np.asarray(p, dtype=float))
            else:
                matched_points.append(np.asarray(pts, dtype=float))

        # Concatenate all dilated coords
        if all_dilated_coords:
            dilated_coords = np.concatenate(all_dilated_coords, axis=0)
        else:
            dilated_coords = np.zeros((0, 3), dtype=int)

        debug_info = {
            "raw_points": raw_points,
            "matched_points": matched_points,
            "dilated_coords": dilated_coords,
            "dilated_shape": vol_shape,
        }
        return preds_cleaned, simplified_pred, debug_info

    return preds_cleaned, simplified_pred
