#!/usr/bin/env python3
"""Build the frozen target-only PRSM donor map for exp375.

This is a pre-metric operation: it reads only paths, labels and pose heatmaps.
No checkpoint, descriptor or retrieval metric is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import cfg as base_cfg  # noqa: E402
from experiments.exp375_prsm.eval_counterfactual import (  # noqa: E402
    WRITE_NUISANCE_NAMES,
    _target_heatmaps,
    constrained_random_bijection,
    matched_nuisance_audit,
    pose_write_nuisance,
)


MAPPING_SEED = 375001
MATCHING_FORMULA = (
    'exp375_target_write_v1=mean(min(abs(robust_z_i-robust_z_j),5));'
    'dims=amplitude+support+xy+part+12row+4column;'
    'hard=split_local,self_neq,pid_neq;float64'
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def robust_scale(values: np.ndarray) -> tuple[np.ndarray, dict]:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    scale = 1.4826 * mad
    active = scale >= 1e-8
    standardized = np.zeros_like(values)
    standardized[:, active] = (
        values[:, active] - median[active]) / scale[active]
    if not np.isfinite(standardized).all() or not active.any():
        raise RuntimeError('invalid robust target-write nuisance matrix')
    return standardized, {
        'median': median.tolist(),
        'mad': mad.tolist(),
        'active_dimensions': np.flatnonzero(active).tolist(),
        'constant_dimensions': np.flatnonzero(~active).tolist(),
    }


def minimum_cost_sparse_mapping(
        standardized: np.ndarray, pids: np.ndarray,
        paths: list[str], device: torch.device,
        topk: int = 32, chunk: int = 16) -> tuple[np.ndarray, dict]:
    """Chunked top-k graph, fast full matching, then deterministic 2-opt."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    values = torch.from_numpy(
        np.asarray(standardized, dtype=np.float64)).to(device)
    pid_tensor = torch.from_numpy(
        np.asarray(pids, dtype=np.int64)).to(device)
    count = len(pids)
    keep = min(int(topk), count - 1)
    if keep < 1:
        raise ValueError('matching split must contain at least two samples')
    path_rank = {
        index: rank for rank, index in enumerate(
            sorted(range(count), key=lambda item: paths[item]))
    }
    lex_rank = torch.tensor(
        [path_rank[index] for index in range(count)],
        dtype=torch.float64, device=device)
    edge_rows = np.empty(count * keep, dtype=np.int32)
    edge_columns = np.empty(count * keep, dtype=np.int32)
    edge_costs = np.empty(count * keep, dtype=np.float64)
    columns = torch.arange(count, device=device).view(1, -1)
    cursor = 0
    for start in range(0, count, int(chunk)):
        stop = min(start + int(chunk), count)
        anchor = values[start:stop]
        costs = (anchor[:, None, :] - values[None, :, :]).abs_()
        costs.clamp_(max=5.0)
        costs = costs.mean(dim=2)
        rows = torch.arange(start, stop, device=device).view(-1, 1)
        eligible = (
            (pid_tensor[start:stop, None] != pid_tensor[None, :])
            & (rows != columns))
        costs[~eligible] = torch.inf
        # A global lexicographic tie-break makes every identical zero-write
        # row choose the same donors and can destroy Hall connectivity.  A
        # row-relative cyclic rank spreads exact ties deterministically while
        # leaving every real nuisance-cost ordering unchanged.
        row_lex_rank = lex_rank[start:stop, None]
        cyclic_rank = torch.remainder(
            lex_rank[None, :] - row_lex_rank, count)
        costs += 1e-12 * cyclic_rank / (count + 1)
        selected_costs, selected = torch.topk(
            costs, k=keep, dim=1, largest=False, sorted=True)
        if not bool(torch.isfinite(selected_costs).all()):
            raise RuntimeError('insufficient eligible target-write donors')
        length = (stop - start) * keep
        edge_rows[cursor:cursor + length] = np.repeat(
            np.arange(start, stop, dtype=np.int32), keep)
        edge_columns[cursor:cursor + length] = selected.cpu().numpy().reshape(-1)
        selected_numpy = selected_costs.cpu().numpy().reshape(-1)
        selected_flat = edge_columns[cursor:cursor + length]
        row_flat = np.repeat(
            np.arange(start, stop, dtype=np.int32), keep)
        selected_ranks = np.fromiter((
            (path_rank[int(column)] - path_rank[int(row)]) % count
            for row, column in zip(row_flat, selected_flat)),
            dtype=np.float64, count=length)
        edge_costs[cursor:cursor + length] = (
            selected_numpy - 1e-12 * selected_ranks / (count + 1))
        cursor += length
        del costs, eligible, selected_costs, selected

    # Keep each row's candidates in increasing-cost order.  Cardinality
    # matching is orders of magnitude faster than SciPy's sparse min-weight
    # solver at gallery scale; every selected edge is still a top-k neighbor.
    donor_matrix = edge_columns[:cursor].reshape(count, keep)
    cost_matrix = edge_costs[:cursor].reshape(count, keep)
    fallback = constrained_random_bijection(pids, MAPPING_SEED)
    fallback_cost = np.minimum(
        np.abs(standardized - standardized[fallback]), 5.0).mean(axis=1)
    graph_donors = np.concatenate(
        [donor_matrix, fallback[:, None]], axis=1)
    graph_costs = np.concatenate(
        [cost_matrix, fallback_cost[:, None]], axis=1)
    graph_width = graph_donors.shape[1]
    graph = csr_matrix((
        np.ones(count * graph_width, dtype=np.uint8),
        graph_donors.reshape(-1),
        np.arange(0, count * graph_width + 1, graph_width,
                  dtype=np.int64)),
        shape=(count, count))
    mapping = maximum_bipartite_matching(
        graph, perm_type='column').astype(np.int64, copy=False)
    if mapping.shape != (count,) or np.any(mapping < 0):
        raise RuntimeError('sparse target-write graph has no full matching')

    # Deterministic pairwise swaps reduce cost without ever leaving the
    # frozen top-k graph or breaking bijectivity.  The scientific acceptance
    # gate below remains the authority; this is only solver engineering.
    edge_cost_lookup = [
        {int(column): float(cost)
         for column, cost in zip(graph_donors[row], graph_costs[row])}
        for row in range(count)
    ]
    inverse = np.empty(count, dtype=np.int64)
    inverse[mapping] = np.arange(count, dtype=np.int64)
    current_costs = np.asarray([
        edge_cost_lookup[row][int(mapping[row])]
        for row in range(count)], dtype=np.float64)
    swap_count = 0
    for _pass in range(4):
        changed = 0
        for row in range(count):
            old_column = int(mapping[row])
            old_cost = float(current_costs[row])
            for candidate, candidate_cost in zip(
                    graph_donors[row], graph_costs[row]):
                candidate = int(candidate)
                other = int(inverse[candidate])
                if other == row:
                    continue
                other_new_cost = edge_cost_lookup[other].get(old_column)
                if other_new_cost is None:
                    continue
                proposed = float(candidate_cost) + other_new_cost
                current = old_cost + float(current_costs[other])
                if proposed + 1e-12 >= current:
                    continue
                other_old_column = int(mapping[other])
                mapping[row] = candidate
                mapping[other] = old_column
                inverse[candidate] = row
                inverse[old_column] = other
                current_costs[row] = float(candidate_cost)
                current_costs[other] = other_new_cost
                old_column = candidate
                old_cost = float(candidate_cost)
                changed += 1
                swap_count += 1
        if changed == 0:
            break
    expected = np.arange(count)
    if (not np.array_equal(np.sort(mapping), expected)
            or np.any(mapping == expected)
            or np.any(pids[mapping] == pids)):
        raise RuntimeError('target-write mapping violates hard constraints')
    matches = graph_donors == mapping[:, None]
    if not bool(matches.any(axis=1).all()):
        raise RuntimeError('solver selected edge outside frozen top-k graph')
    mapped_costs = graph_costs[
        np.arange(count), matches.argmax(axis=1)]
    outside_topk = ~(
        donor_matrix == mapping[:, None]).any(axis=1)
    return mapping, {
        'edge_count': int(count * graph_width),
        'topk': keep,
        'connectivity_fallback': 'one_constrained_random_bijection_edge_per_row',
        'outside_topk_edges_used': int(outside_topk.sum()),
        'solver': 'maximum_bipartite_matching+deterministic_2opt',
        'cost_improving_swaps': int(swap_count),
        'mean_graph_cost': float(mapped_costs.mean()),
        'p95_graph_cost': float(np.quantile(mapped_costs, 0.95)),
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-file', type=Path,
        default=ROOT / 'configs/occluded_duke/exp375_p0_prsm.yml')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--topk', type=int, default=32)
    return parser.parse_args()


def main() -> int:
    from datasets import make_dataloader
    from model.modules.pose_routed_selective_memory import (
        PoseRoutedSelectiveMemory,
    )

    args = _parse_args()
    if args.topk < 2:
        raise ValueError('--topk must be at least 2')
    matching_formula = (
        MATCHING_FORMULA + ';topk=%d;fallback=one_constrained_random_edge_per_row'
        % int(args.topk))
    cfg = base_cfg.clone()
    cfg.merge_from_file(str(args.config_file))
    cfg.freeze()
    seed = int(cfg.SOLVER.SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)

    (_, _, loader, num_query, _num_classes,
     _camera_num, _view_num) = make_dataloader(cfg)
    module = PoseRoutedSelectiveMemory(
        feat_dim=768, routing='parts').to(device).eval()
    nuisance_batches = []
    metadata = []
    with torch.inference_mode():
        for batch in loader:
            (_images, pids, camids_eval, _camids, _viewids,
             imgpaths, pose_dict) = batch
            target = _target_heatmaps(pose_dict).to(device)
            nuisance_batches.append(
                pose_write_nuisance(module, target).cpu())
            for path, pid, camid in zip(imgpaths, pids, camids_eval):
                metadata.append({
                    'path': str(Path(path).resolve()),
                    'pid': int(pid), 'camid': int(camid),
                })
    nuisance = torch.cat(nuisance_batches).numpy().astype(np.float64)
    if nuisance.shape != (len(loader.dataset), len(WRITE_NUISANCE_NAMES)):
        raise RuntimeError('target-write nuisance count mismatch')

    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.output_dir / 'target_write_nuisance.npy', nuisance)
    mappings = {}
    split_reports = {}
    split_specs = (
        ('query', 0, num_query),
        ('gallery', num_query, len(metadata)),
    )
    for split, start, stop in split_specs:
        split_values = nuisance[start:stop]
        standardized, scaler = robust_scale(split_values)
        split_metadata = []
        for local, row in enumerate(metadata[start:stop]):
            split_metadata.append({
                'split': split, 'index': local, **row,
            })
        pids = np.asarray(
            [row['pid'] for row in split_metadata], dtype=np.int64)
        paths = [row['path'] for row in split_metadata]
        mapping, graph_report = minimum_cost_sparse_mapping(
            standardized, pids, paths, device, topk=args.topk)
        mappings[split] = mapping
        np.save(
            args.output_dir / f'{split}_mappings.npy',
            mapping[None].astype(np.int32))
        (args.output_dir / f'{split}_metadata.json').write_text(
            json.dumps(split_metadata, ensure_ascii=False,
                       sort_keys=True, separators=(',', ':')) + '\n',
            encoding='utf-8')
        split_audit = matched_nuisance_audit(
            torch.from_numpy(split_values).float(),
            torch.from_numpy(split_values[mapping]).float(),
            mapping, loader.dataset.dataset[start:stop],
            num_query=stop - start)
        mapping_audit = {
            'cost_formula_version': matching_formula,
            'mapping_audits': [split_audit],
            'mapping_seeds': [MAPPING_SEED],
            'solver': {
                'name': graph_report['solver'],
                'topk': graph_report['topk'],
            },
            'effective_unique_count': 1,
            'scaler': scaler,
            'graph': graph_report,
        }
        (args.output_dir / f'{split}_mapping_audit.json').write_text(
            json.dumps(mapping_audit, ensure_ascii=False, indent=2,
                       sort_keys=True) + '\n', encoding='utf-8')
        split_reports[split] = split_audit

    combined_mapping = np.concatenate([
        mappings['query'], mappings['gallery'] + num_query])
    combined_audit = matched_nuisance_audit(
        torch.from_numpy(nuisance).float(),
        torch.from_numpy(nuisance[combined_mapping]).float(),
        combined_mapping, loader.dataset.dataset, num_query)
    manifest = {
        'status': 'PASS',
        'schema': 'exp375_target_write_mapping_v1',
        'seed': MAPPING_SEED,
        'formula': matching_formula,
        'config': str(args.config_file.resolve()),
        'nuisance_names': list(WRITE_NUISANCE_NAMES),
        'split_audits': split_reports,
        'combined_audit': combined_audit,
    }
    for path in sorted(args.output_dir.iterdir()):
        manifest.setdefault('artifact_sha256', {})[path.name] = _sha256(path)
    (args.output_dir / 'MANIFEST.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2,
                   sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({
        'status': 'PASS',
        'output_dir': str(args.output_dir.resolve()),
        'query_ratio': split_reports['query'][
            'mean_cost_over_random_median'],
        'gallery_ratio': split_reports['gallery'][
            'mean_cost_over_random_median'],
        'combined_ratio': combined_audit[
            'mean_cost_over_random_median'],
    }, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
