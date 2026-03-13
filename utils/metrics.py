import torch
import numpy as np
import os
import torch.nn.functional as F
from utils.reranking import re_ranking


def nfc(feat, k1=2, k2=2):
    """Neighbor Feature Centralization (from Pose2ID, CVPR 2025).

    For each feature, find mutual top-k nearest neighbors and add their
    features.  This is a test-time feature augmentation that does not
    require any training.

    Args:
        feat: (N, C) tensor of features
        k1: number of top-k neighbors to consider
        k2: mutual verification depth
    Returns:
        (N, C) augmented features
    """
    feat = feat.clone()
    # Compute pairwise squared euclidean distance
    m = feat.shape[0]
    dist = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(m, m) + \
           torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(m, m).t()
    dist.addmm_(feat, feat.t(), beta=1, alpha=-2)

    # Set self-distance to large value
    eye = torch.eye(m, device=dist.device)
    dist[eye == 1] = 1e6

    # Find top-k nearest neighbors
    _, rank = dist.topk(max(k1, k2), largest=False)

    # Build mutual top-k lists
    mutual_topk_list = []
    for i in range(m):
        mutual_list = []
        for j in rank[i, :k1]:
            if i in rank[j.item(), :k2]:
                mutual_list.append(j.item())
        mutual_topk_list.append(mutual_list)

    # Add mutual neighbor features
    feat_copy = feat.clone()
    for i in range(m):
        if mutual_topk_list[i]:
            feat[i] += feat_copy[mutual_topk_list[i]].sum(dim=0)

    return feat


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
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False, cfg=None):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.cfg = cfg
        # NFC (Neighbor Feature Centralization) config
        self.nfc_enabled = False
        self.nfc_k1 = 2
        self.nfc_k2 = 2
        if cfg is not None:
            self.nfc_enabled = getattr(cfg.TEST, 'NFC', False)
            self.nfc_k1 = getattr(cfg.TEST, 'NFC_K1', 2)
            self.nfc_k2 = getattr(cfg.TEST, 'NFC_K2', 2)

    def reset(self):
        self.feats = []
        self.structured_feats = None
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        if isinstance(feat, dict):
            if self.structured_feats is None:
                self.structured_feats = {
                    'mode': feat['mode'],
                    'global_feat': [],
                    'kp_feats': [],
                    'kp_weights': [],
                }
            self.structured_feats['global_feat'].append(feat['global_feat'].cpu())
            self.structured_feats['kp_feats'].append(feat['kp_feats'].cpu())
            self.structured_feats['kp_weights'].append(feat['kp_weights'].cpu())
        else:
            self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        if self.structured_feats is not None:
            return self._compute_structured()

        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel

        # Apply NFC if enabled (before splitting query/gallery)
        if self.nfc_enabled:
            print(f'=> Applying NFC (k1={self.nfc_k1}, k2={self.nfc_k2})')
            feats = nfc(feats.cuda(), k1=self.nfc_k1, k2=self.nfc_k2).cpu()
            # Re-normalize after NFC augmentation
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf

    def _compute_structured(self):
        global_feats = torch.cat(self.structured_feats['global_feat'], dim=0)
        kp_feats = torch.cat(self.structured_feats['kp_feats'], dim=0)
        kp_weights = torch.cat(self.structured_feats['kp_weights'], dim=0)
        mode = self.structured_feats['mode']

        if self.feat_norm:
            print("The test feature is normalized")
            global_feats = F.normalize(global_feats, dim=1, p=2)
            kp_feats = F.normalize(kp_feats, dim=2, p=2)

        q_global = global_feats[:self.num_query]
        g_global = global_feats[self.num_query:]
        q_kp = kp_feats[:self.num_query]
        g_kp = kp_feats[self.num_query:]
        q_w = kp_weights[:self.num_query]
        g_w = kp_weights[self.num_query:]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        print('=> Computing DistMat with common-visible keypoint reasoning')
        global_dist = self._euclidean_distance_tensor(q_global, g_global)
        kp_dist = self._common_visible_kp_distance(q_kp, g_kp, q_w, g_w, global_dist, mode)

        if mode == 'cvk_only':
            distmat = kp_dist.cpu().numpy()
        else:
            gw = float(getattr(self.cfg.TEST, 'CVK_GLOBAL_WEIGHT', 1.0)) if self.cfg is not None else 1.0
            kw = float(getattr(self.cfg.TEST, 'CVK_KP_WEIGHT', 1.0)) if self.cfg is not None else 1.0
            distmat = ((gw * global_dist + kw * kp_dist) / max(gw + kw, 1e-12)).cpu().numpy()

        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        return cmc, mAP, distmat, self.pids, self.camids, q_global, g_global

    @staticmethod
    def _euclidean_distance_tensor(qf, gf):
        m = qf.shape[0]
        n = gf.shape[0]
        dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
        return dist_mat.clamp_min_(0.0)

    def _common_visible_kp_distance(self, q_kp, g_kp, q_w, g_w, global_dist, mode):
        q_kp_t = q_kp.transpose(1, 0)  # (K, Q, C)
        g_kp_t = g_kp.transpose(1, 0)  # (K, G, C)
        dot = torch.matmul(q_kp_t, g_kp_t.transpose(2, 1))
        q_sq = q_kp_t.pow(2).sum(dim=-1)
        g_sq = g_kp_t.pow(2).sum(dim=-1)
        kp_dist = (q_sq.unsqueeze(2) - 2 * dot + g_sq.unsqueeze(1)).clamp_min_(0.0).sqrt_()

        weights = torch.sqrt(
            q_w.transpose(1, 0).unsqueeze(2) * g_w.transpose(1, 0).unsqueeze(1)
        )
        weight_sum = weights.sum(dim=0)
        masked = (kp_dist * weights).sum(dim=0) / weight_sum.clamp(min=1e-12)

        if mode == 'cvk_only':
            fallback = kp_dist.max().detach() + 1.0
            masked = torch.where(weight_sum > 0, masked, torch.full_like(masked, fallback))
        else:
            masked = torch.where(weight_sum > 0, masked, global_dist)
        return masked


