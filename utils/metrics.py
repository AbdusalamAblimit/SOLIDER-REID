import torch
import numpy as np
import os
from utils.reranking import re_ranking

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalResultItem:
    """封装单个分支的评估结果，默认只携带指标，避免保存大矩阵占内存。"""

    cmc: np.ndarray
    mAP: float
    distmat: Optional[np.ndarray] = None
    pids: Optional[np.ndarray] = None
    camids: Optional[np.ndarray] = None
    qf: Optional[torch.Tensor] = None
    gf: Optional[torch.Tensor] = None

def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False, num_gallery: Optional[int] = None,
                 dist_chunk_size: int = 1024):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.num_gallery = num_gallery
        self.dist_chunk_size = max(1, int(dist_chunk_size))
        self.reset()

    def reset(self):
        self._feat_buffers = defaultdict(list)
        self.pids = []
        self.camids = []
        self._num_processed_query = 0
        self._num_processed_gallery = 0

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        if isinstance(feat, dict):
            for key, value in feat.items():
                self._feat_buffers[key].append(value.detach().cpu())
        elif isinstance(feat, (list, tuple)):
            for idx, value in enumerate(feat):
                self._feat_buffers[f'branch{idx}'].append(value.detach().cpu())
        else:
            self._feat_buffers['global'].append(feat.detach().cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

        batch_size = self._infer_batch_size(feat)
        if self.num_query > 0:
            remaining_query = max(self.num_query - self._num_processed_query, 0)
        else:
            remaining_query = 0
        query_in_batch = min(batch_size, remaining_query)
        gallery_in_batch = batch_size - query_in_batch

        if query_in_batch > 0:
            self._num_processed_query += query_in_batch
            print(f"[Eval] Extracted query features: {self._num_processed_query}/{self.num_query}")

        if gallery_in_batch > 0:
            self._num_processed_gallery += gallery_in_batch
            if self.num_gallery is not None:
                print(f"[Eval] Extracted gallery features: {self._num_processed_gallery}/{self.num_gallery}")
            else:
                print(f"[Eval] Extracted gallery features: {self._num_processed_gallery}")

    def compute(self, keep_details: bool = False, clear_buffers: bool = True):  # called after each epoch
        if not self._feat_buffers:
            raise RuntimeError('No features to evaluate.')
        results = OrderedDict()
        pids = np.asarray(self.pids)
        camids = np.asarray(self.camids)
        for key, feat_list in self._feat_buffers.items():
            feats = torch.cat(feat_list, dim=0)
            if self.feat_norm:
                print(f"The test feature ({key}) is normalized")
                feats = torch.nn.functional.normalize(feats, dim=1, p=2)
            qf = feats[:self.num_query]
            gf = feats[self.num_query:]
            q_pids = pids[:self.num_query]
            g_pids = pids[self.num_query:]
            q_camids = camids[:self.num_query]
            g_camids = camids[self.num_query:]

            if self.reranking:
                print('=> Enter reranking')
                distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            else:
                print(f'=> Computing DistMat with euclidean_distance ({key})')
                distmat = self._chunked_euclidean_distance(qf, gf)
            cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
            extra_kwargs = {}
            if keep_details:
                # 需要导出距离矩阵等信息时才回传这些大对象，默认情况下避免占用大量内存
                extra_kwargs = dict(
                    distmat=distmat,
                    pids=pids.copy(),
                    camids=camids.copy(),
                    qf=qf,
                    gf=gf,
                )
            results[key] = EvalResultItem(cmc=cmc, mAP=mAP, **extra_kwargs)

        if clear_buffers:
            # 默认在计算完一次评估后立即释放缓存，避免这些特征在下一轮验证前继续占用显存/内存
            self.reset()

        if len(results) == 1:
            return next(iter(results.values()))
        return results

    @staticmethod
    def _infer_batch_size(feat) -> int:
        if isinstance(feat, dict):
            if not feat:
                return 0
            sample = next(iter(feat.values()))
        elif isinstance(feat, (list, tuple)):
            if not feat:
                return 0
            sample = feat[0]
        else:
            sample = feat
        if not torch.is_tensor(sample):
            raise TypeError(f"Unexpected feature type for batch size inference: {type(sample)}")
        return sample.shape[0]

    def _chunked_euclidean_distance(self, qf: torch.Tensor, gf: torch.Tensor) -> np.ndarray:
        if qf.numel() == 0 or gf.numel() == 0:
            return torch.empty((qf.shape[0], gf.shape[0]), dtype=qf.dtype).cpu().numpy()

        q_chunk = min(self.dist_chunk_size, max(1, qf.shape[0]))
        g_chunk = min(self.dist_chunk_size, max(1, gf.shape[0]))
        q_starts = list(range(0, qf.shape[0], q_chunk))
        g_starts = list(range(0, gf.shape[0], g_chunk))
        total_q_chunks = len(q_starts)

        print(f"[Eval] Distance matrix will be computed in {total_q_chunks}x{len(g_starts)} chunks.")
        distmat = torch.empty((qf.shape[0], gf.shape[0]), dtype=qf.dtype)
        for qi, q_start in enumerate(q_starts):
            q_end = min(q_start + q_chunk, qf.shape[0])
            q_batch = qf[q_start:q_end]
            q_norm = torch.pow(q_batch, 2).sum(dim=1, keepdim=True)
            for g_start in g_starts:
                g_end = min(g_start + g_chunk, gf.shape[0])
                g_batch = gf[g_start:g_end]
                g_norm = torch.pow(g_batch, 2).sum(dim=1).view(1, -1)
                block = q_norm + g_norm
                block.addmm_(q_batch, g_batch.t(), beta=1.0, alpha=-2.0)
                distmat[q_start:q_end, g_start:g_end] = block
            print(f"[Eval] Distance progress: {qi + 1}/{total_q_chunks} query chunks processed")

        return distmat.cpu().numpy()




