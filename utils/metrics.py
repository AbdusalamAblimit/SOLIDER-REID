import torch
import numpy as np
import os
import torch.nn.functional as F
from utils.reranking import re_ranking
from model.modules.pair_adaptive_fusion import (
    build_pair_descriptors,
    build_query_competition_descriptors,
    build_query_context_descriptors,
    common_support_distance,
    euclidean_distance_tensor,
)


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
        self.pair_fusion_head = None
        self.pair_residual_head = None
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
        # Power normalization: sign(x) * |x|^alpha — suppresses burstiness in structural tokens
        power_alpha = float(getattr(self.cfg.TEST, 'POWER_NORM', 0.0)) if self.cfg else 0.0
        if power_alpha > 0:
            print(f"Applying power normalization (alpha={power_alpha})")
            feats = torch.sign(feats) * torch.abs(feats).pow(power_alpha)
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

        # MaxSim modes: ColBERT-style late interaction
        if mode in ('maxsim', 'maxsim_hybrid'):
            print(f'=> MaxSim mode: {mode}')
            maxsim_dist = self._maxsim_distance(q_kp, g_kp, q_w, g_w)
            if mode == 'maxsim':
                distmat = maxsim_dist.cpu().numpy()
            else:  # maxsim_hybrid
                gw = float(getattr(self.cfg.TEST, 'CVK_GLOBAL_WEIGHT', 1.0)) if self.cfg is not None else 1.0
                kw = float(getattr(self.cfg.TEST, 'CVK_KP_WEIGHT', 1.0)) if self.cfg is not None else 1.0
                distmat = ((gw * global_dist + kw * maxsim_dist) / max(gw + kw, 1e-12)).cpu().numpy()
            cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
            return cmc, mAP, distmat, self.pids, self.camids, q_global, g_global

        kp_dist, support_ratio = self._common_visible_kp_distance(
            q_kp, g_kp, q_w, g_w, global_dist, mode, return_ratio=True)

        if mode == 'cvk_only':
            distmat = kp_dist.cpu().numpy()
        elif mode == 'cvk_adaptive':
            if self.pair_fusion_head is None:
                raise ValueError('POSE_TEST_FEAT=cvk_adaptive requires evaluator.pair_fusion_head')
            head = self.pair_fusion_head
            head_was_training = head.training
            head.eval()
            head_device = next(head.parameters()).device
            g_vis_mean = g_w.mean(dim=1).unsqueeze(0)
            chunk_size = 256
            mixed_chunks = []
            with torch.no_grad():
                for start in range(0, q_w.shape[0], chunk_size):
                    end = min(q_w.shape[0], start + chunk_size)
                    q_vis_mean = q_w[start:end].mean(dim=1, keepdim=True).expand(-1, g_w.shape[0])
                    desc = build_pair_descriptors(
                        global_dist[start:end],
                        kp_dist[start:end],
                        support_ratio[start:end],
                        q_vis_mean,
                        g_vis_mean.expand(end - start, -1),
                    )
                    alpha = head(desc.to(head_device).view(-1, desc.shape[-1]))
                    alpha = alpha.view(end - start, desc.shape[1]).to(global_dist.device)
                    mixed = (1.0 - alpha) * global_dist[start:end] + alpha * kp_dist[start:end]
                    mixed_chunks.append(mixed.cpu())
            if head_was_training:
                head.train()
            distmat = torch.cat(mixed_chunks, dim=0).numpy()
        elif mode == 'cvk_residual':
            if self.pair_residual_head is None:
                raise ValueError('POSE_TEST_FEAT=cvk_residual requires evaluator.pair_residual_head')
            head = self.pair_residual_head
            head_was_training = head.training
            head.eval()
            head_device = next(head.parameters()).device
            lpcs_head_mode = getattr(self.cfg.MODEL, 'POSE_LPCS_HEAD_MODE', 'residual') if self.cfg is not None else 'residual'
            gw = float(getattr(self.cfg.TEST, 'CVK_GLOBAL_WEIGHT', 1.0)) if self.cfg is not None else 1.0
            kw = float(getattr(self.cfg.TEST, 'CVK_KP_WEIGHT', 1.0)) if self.cfg is not None else 1.0
            base_dist = (gw * global_dist + kw * kp_dist) / max(gw + kw, 1e-12)
            g_vis_mean = g_w.mean(dim=1).unsqueeze(0)
            chunk_size = 256
            corrected_chunks = []
            with torch.no_grad():
                for start in range(0, q_w.shape[0], chunk_size):
                    end = min(q_w.shape[0], start + chunk_size)
                    q_vis_mean = q_w[start:end].mean(dim=1, keepdim=True).expand(-1, g_w.shape[0])
                    desc = build_pair_descriptors(
                        global_dist[start:end],
                        kp_dist[start:end],
                        support_ratio[start:end],
                        q_vis_mean,
                        g_vis_mean.expand(end - start, -1),
                    )
                    context_mode = getattr(self.cfg.MODEL, 'POSE_LPCS_CONTEXT_MODE', 'none')
                    if context_mode == 'query_ctx':
                        row_ctx = build_query_context_descriptors(
                            base_dist[start:end],
                            support_ratio[start:end],
                            pair_change=(kp_dist[start:end] - base_dist[start:end]).abs(),
                        )
                        desc = torch.cat([desc, row_ctx], dim=-1)
                    elif context_mode == 'comp_ctx':
                        comp_ctx = build_query_competition_descriptors(
                            base_dist[start:end],
                            kp_dist[start:end],
                            support_ratio[start:end],
                        )
                        desc = torch.cat([desc, comp_ctx], dim=-1)
                    if lpcs_head_mode == 'residual_conf':
                        raw_delta, conf_logits = head(desc.to(head_device).view(-1, desc.shape[-1]))
                        raw_delta = raw_delta.view(end - start, desc.shape[1]).to(base_dist.device)
                        conf_logits = conf_logits.view(end - start, desc.shape[1]).to(base_dist.device)
                        conf = torch.sigmoid(conf_logits)
                        delta = conf * raw_delta
                    else:
                        delta = head(desc.to(head_device).view(-1, desc.shape[-1]))
                        delta = delta.view(end - start, desc.shape[1]).to(base_dist.device)
                    corrected = base_dist[start:end] + delta
                    corrected_chunks.append(corrected.cpu())
            if head_was_training:
                head.train()
            distmat = torch.cat(corrected_chunks, dim=0).numpy()
        else:
            gw = float(getattr(self.cfg.TEST, 'CVK_GLOBAL_WEIGHT', 1.0)) if self.cfg is not None else 1.0
            kw = float(getattr(self.cfg.TEST, 'CVK_KP_WEIGHT', 1.0)) if self.cfg is not None else 1.0
            distmat = ((gw * global_dist + kw * kp_dist) / max(gw + kw, 1e-12)).cpu().numpy()

        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        return cmc, mAP, distmat, self.pids, self.camids, q_global, g_global

    @staticmethod
    def _euclidean_distance_tensor(qf, gf):
        return euclidean_distance_tensor(qf, gf)

    @staticmethod
    def _maxsim_distance(q_kp, g_kp, q_w, g_w, vis_thr=0.3):
        """ColBERT-style MaxSim distance for keypoint sets.

        For each query keypoint, find the most similar gallery keypoint,
        weighted by query keypoint confidence. Naturally handles occlusion:
        low-quality keypoints produce low max-similarity and contribute less.

        Args:
            q_kp: (Q, 17, D) query keypoint features (already L2-normed)
            g_kp: (G, 17, D) gallery keypoint features (already L2-normed)
            q_w: (Q, 17) query keypoint weights
            g_w: (G, 17) gallery keypoint weights
            vis_thr: minimum weight to consider a keypoint "active"
        Returns:
            dist: (Q, G) distance matrix (lower = more similar)
        """
        Q, K, D = q_kp.shape
        G = g_kp.shape[0]
        device = q_kp.device

        # Process in chunks to avoid OOM
        chunk_size = 128
        dist_chunks = []
        for qi in range(0, Q, chunk_size):
            qe = min(qi + chunk_size, Q)
            q_batch = q_kp[qi:qe]  # (B, 17, D)
            w_batch = q_w[qi:qe]   # (B, 17)

            chunk_dists = []
            for gi in range(0, G, chunk_size):
                ge = min(gi + chunk_size, G)
                g_batch = g_kp[gi:ge]  # (C, 17, D)

                # Cosine similarity: (B, 17_q, C, 17_g)
                # Efficient: (B, 17, D) @ (C, 17, D).T → (B, 17, C, 17)
                sim = torch.einsum('bkd,cjd->bkcj', q_batch, g_batch)

                # For each query keypoint, max over gallery keypoints
                # (B, 17_q, C)
                maxsim_per_qkp = sim.max(dim=3)[0]

                # Weight by query keypoint confidence
                # Mask out low-confidence query keypoints
                w_mask = (w_batch > vis_thr).float()  # (B, 17)
                w_eff = w_batch * w_mask  # (B, 17)
                w_sum = w_eff.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)

                # Weighted average MaxSim: (B, C)
                weighted_maxsim = (maxsim_per_qkp * w_eff.unsqueeze(2)).sum(dim=1) / w_sum

                # Convert similarity to distance
                chunk_dists.append(1.0 - weighted_maxsim)

            dist_chunks.append(torch.cat(chunk_dists, dim=1))

        return torch.cat(dist_chunks, dim=0)

    def _common_visible_kp_distance(self, q_kp, g_kp, q_w, g_w, global_dist, mode, return_ratio=False):
        if mode == 'cvk_only':
            fallback = global_dist.max().detach() + 1.0
            masked, support_ratio = common_support_distance(
                q_kp, g_kp, q_w, g_w, fallback=torch.full_like(global_dist, fallback), return_ratio=True)
        else:
            masked, support_ratio = common_support_distance(
                q_kp, g_kp, q_w, g_w, fallback=global_dist, return_ratio=True)
        if return_ratio:
            return masked, support_ratio
        return masked
