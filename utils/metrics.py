import torch
import numpy as np
import os
from utils.reranking import re_ranking


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
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        # For VPReID dict features
        self.has_parts = False
        self.part_feats = []
        self.part_vis_list = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output

        if isinstance(feat, dict):
            # VPReID dict: {'global': [B, D], 'parts': [B, K, D], 'part_vis': [B, K]}
            self.has_parts = True
            self.feats.append(feat['global'].cpu())
            if 'parts' in feat:
                self.part_feats.append(feat['parts'].cpu())
            if 'part_vis' in feat:
                self.part_vis_list.append(feat['part_vis'].cpu())
        else:
            self.feats.append(feat.cpu())

        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel

        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        # Global distance
        qf = feats[:self.num_query]
        gf = feats[self.num_query:]

        if self.reranking:
            print('=> Enter reranking')
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)

        cmc_global, mAP_global = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        # If we have part features, also compute visibility-weighted part distance
        if self.has_parts and self.part_feats:
            parts = torch.cat(self.part_feats, dim=0)  # [N, K, D]
            vis = torch.cat(self.part_vis_list, dim=0) if self.part_vis_list else None  # [N, K]

            if self.feat_norm:
                B, K, D = parts.shape
                parts = parts.view(B, K * D)
                parts = torch.nn.functional.normalize(parts, dim=1, p=2)
                parts = parts.view(B, K, D)

            q_parts = parts[:self.num_query]
            g_parts = parts[self.num_query:]
            q_vis = vis[:self.num_query] if vis is not None else None
            g_vis = vis[self.num_query:] if vis is not None else None

            # Part distance: concat all part features (simple but effective)
            qf_parts = q_parts.view(q_parts.shape[0], -1)
            gf_parts = g_parts.view(g_parts.shape[0], -1)
            distmat_parts = euclidean_distance(qf_parts, gf_parts)
            cmc_parts, mAP_parts = eval_func(distmat_parts, q_pids, g_pids, q_camids, g_camids)

            # Fused distance: weighted combination of global and part
            distmat_fused = 0.5 * distmat + 0.5 * distmat_parts
            cmc_fused, mAP_fused = eval_func(distmat_fused, q_pids, g_pids, q_camids, g_camids)

            print(f'=> [global] mAP: {mAP_global:.1%}, R-1: {cmc_global[0]:.1%}')
            print(f'=> [parts]  mAP: {mAP_parts:.1%}, R-1: {cmc_parts[0]:.1%}')
            print(f'=> [fused]  mAP: {mAP_fused:.1%}, R-1: {cmc_fused[0]:.1%}')

            # Return best result (fused)
            return cmc_fused, mAP_fused, distmat_fused, self.pids, self.camids, qf, gf

        return cmc_global, mAP_global, distmat, self.pids, self.camids, qf, gf



