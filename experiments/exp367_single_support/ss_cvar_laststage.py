#!/usr/bin/env python3
"""Single-Support CVaR last-stage FT — codex 路径 #2 (解冻 swin base.stages[-1] 真验).

frozen head 两轮 cvar≈random → 解冻 last stage 改特征, 真验 CVaR 是否有用。
基于 codex_laststage_design.md 骨架(codex 调研项目接口)。3 mode 对照:
  cvar  : base CE+triplet + lam·(ss_ce + CVaR_α(support risk))
  random: base + ss_ce (无 CVaR, 证不是 episode 本身涨)
  plain : base only (普通 continued FT, 证不是 last-stage FT 本身涨)
评估: full-gallery + single-support random/worst diagnostic(复用 probe v2 口径)。
backbone 训练, codex 三审 diff 后跑。
"""
import os, sys, math, time, random, argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root for config/model/datasets/loss/solver/processor
from config import cfg
from datasets import make_dataloader
from model import make_model
from loss import make_loss
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from processor.processor import _pose_to_device, _extract_feat_flip


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser("Single-Support CVaR last-stage FT")
    p.add_argument("--config_file", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", choices=["cvar", "random", "plain"], default="cvar")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--lam", type=float, default=0.3)
    p.add_argument("--ss_weight", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--eval_period", type=int, default=5)
    p.add_argument("--eval_seeds", type=int, default=20)
    p.add_argument("--flip_test", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("opts", nargs=argparse.REMAINDER)
    return p.parse_args()


def cfg_setup(args):
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.defrost()
    cfg.OUTPUT_DIR = args.output_dir
    cfg.SOLVER.MAX_EPOCHS = args.epochs
    cfg.TEST.FLIP_TEST = bool(args.flip_test)
    cfg.TEST.NECK_FEAT = "after"   # codex High: 对齐 probe v2 (BNNeck after global)
    assert 0.0 < args.alpha < 1.0 and args.tau > 0, "alpha in (0,1), tau>0"  # codex Low guard
    # 关 POSE aug 保证 train batch 是干净 P×K (不是 3/4-view augmentation)
    for k in ["POSE_PARALLEL_AUG", "POSE_OA_SD", "POSE_OA_RD", "POSE_PCVT"]:
        if k in cfg.MODEL:
            setattr(cfg.MODEL, k, False)
    cfg.freeze()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


def unwrap_out(out):
    if not isinstance(out, (tuple, list)):
        raise RuntimeError(f"unexpected model output type: {type(out)}")
    n = len(out)
    if n == 5: return out[0], out[1], out[2], out[3], out[4]
    if n == 4: return out[0], out[1], out[2], out[3], None
    if n == 3: return out[0], out[1], out[2], None, None
    if n == 2: return out[0], out[1], None, None, None
    raise RuntimeError(f"unexpected model output len={n}")


def global_train_feat(feat):
    return feat[0] if isinstance(feat, (list, tuple)) else feat


def unfreeze_last_stage(model):
    m = model.module if hasattr(model, "module") else model
    for p in m.parameters():
        p.requires_grad = False
    if not hasattr(m, "base") or not hasattr(m.base, "stages"):
        raise RuntimeError("model.base.stages not found; expects Swin backbone")
    for p in m.base.stages[-1].parameters():
        p.requires_grad = True
    if hasattr(m.base, "norm3"):                  # codex: Swin 输出 LayerNorm 在 stages 外, 一并解冻成完整 last-stage 输出层
        for p in m.base.norm3.parameters():
            p.requires_grad = True
    for name in ["bottleneck", "classifier"]:
        if hasattr(m, name):
            for p in getattr(m, name).parameters():
                p.requires_grad = True
    if hasattr(m, "bottleneck") and getattr(m.bottleneck, "bias", None) is not None:
        m.bottleneck.bias.requires_grad_(False)   # codex: 保持原模型 BNNeck bias 冻结
    tr = [(n, p.numel()) for n, p in m.named_parameters() if p.requires_grad]
    print(f"[unfreeze] trainable tensors={len(tr)} params={sum(x[1] for x in tr):,}", flush=True)


def set_frozen_bn_eval(model):
    for mod in model.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            if not any(p.requires_grad for p in mod.parameters(recurse=False)):
                mod.eval()


def add_aux_pose_losses(loss, cfg, kp_data, recon_loss):
    if recon_loss is not None:
        loss = loss + recon_loss
    if not isinstance(kp_data, dict):
        return loss
    if kp_data.get("assign_loss") is not None:
        if getattr(cfg.MODEL, "POSE_LGPA", False):
            w = float(getattr(cfg.MODEL, "POSE_LGPA_ASSIGN_WEIGHT", 0.5))
        elif getattr(cfg.MODEL, "POSE_PPA", False):
            w = float(getattr(cfg.MODEL, "POSE_PPA_ASSIGN_WEIGHT", 0.5))
        else:
            w = 0.5
        loss = loss + w * kp_data["assign_loss"]
    if kp_data.get("clip_id_loss") is not None:
        loss = loss + float(getattr(cfg.MODEL, "POSE_CLIP_ID_WEIGHT", 1.0)) * kp_data["clip_id_loss"]
    if kp_data.get("fsdc_loss") is not None:
        loss = loss + float(getattr(cfg.MODEL, "POSE_FSDC_WEIGHT", 0.5)) * kp_data["fsdc_loss"]
    return loss


def ss_episode_loss(feat, labels, K, alpha, tau):
    if K < 2:
        raise RuntimeError("single-support CVaR needs NUM_INSTANCE >= 2")
    B, D = feat.shape
    if B % K != 0:
        raise RuntimeError(f"batch {B} not divisible by K={K}")
    P = B // K
    lab = labels.view(P, K)
    if not torch.equal(lab, lab[:, :1].expand_as(lab)):
        raise RuntimeError("batch not grouped as P contiguous ids × K instances")
    z = F.normalize(feat.float(), dim=1).contiguous().view(P, K, D)  # codex: AMP 内 float 防半精度 normalize NaN
    risks = []
    for s in range(K):
        proto = z[:, s]
        q_slots = [j for j in range(K) if j != s]
        qz = z[:, q_slots].reshape(P, K - 1, D)
        logits = torch.einsum("nqd,md->nqm", qz, proto) / tau
        tgt = torch.arange(P, device=feat.device)[:, None].expand(P, K - 1)
        ce = F.cross_entropy(logits.reshape(-1, P), tgt.reshape(-1), reduction="none").view(P, K - 1)
        risks.append(ce.mean(1))
    risks = torch.stack(risks, dim=1)            # [P,K]
    ss_ce = risks.mean()
    tail_k = max(1, int(math.ceil((1.0 - alpha) * K)))
    ss_cvar = torch.topk(risks, tail_k, dim=1).values.mean()
    return ss_ce, ss_cvar


def call_project_loss(loss_func, score, feat, target, cam, kp_data):
    # codex: make_loss 支持 kp_data, 直接调用(不 try/except 静默降级掩盖 bug)
    return loss_func(score, feat, target, cam, kp_data=kp_data)


@torch.no_grad()
def extract_global_eval(cfg, model, val_loader, num_query, device, flip_test=False):
    m = model.module if hasattr(model, "module") else model
    old = getattr(m, "pose_test_feat", None)
    if old is not None:
        m.pose_test_feat = "global"
    model.eval()
    feats, pids, camids_all = [], [], []
    for batch in val_loader:
        if cfg.MODEL.POSE_ENABLED:
            img, pid, camid, camids, viewid, _, pose_dict = batch
            pose_dict = _pose_to_device(pose_dict, device)
        else:
            img, pid, camid, camids, viewid, _ = batch
            pose_dict = None
        img = img.to(device); camids = camids.to(device); viewid = viewid.to(device)
        # codex High: eval 强制 pose-off (pose_dict=None, pose_enabled=False), 对齐 probe v2 纯 backbone global
        feat = _extract_feat_flip(model, img, None, camids, viewid, False, flip_test)
        if isinstance(feat, dict):
            raise RuntimeError("expected global tensor feat, got dict")
        feats.append(feat.detach().cpu())
        pids.extend(np.asarray(pid)); camids_all.extend(np.asarray(camid))
    if old is not None:
        m.pose_test_feat = old
    feats = F.normalize(torch.cat(feats, 0).float(), dim=1).numpy()
    pids = np.asarray(pids); camids_all = np.asarray(camids_all)
    return (feats[:num_query], pids[:num_query], camids_all[:num_query],
            feats[num_query:], pids[num_query:], camids_all[num_query:])


def eval_fixed(qf, qp, qc, gf, gp, gc, g_idx, valid_q):
    gff, gpp, gcc = gf[g_idx], gp[g_idx], gc[g_idx]
    aps, r1s, false10 = [], [], []
    for i in valid_q:
        sim = qf[i] @ gff.T
        keep = ~((gpp == qp[i]) & (gcc == qc[i]))
        s, gpk = sim[keep], gpp[keep]
        order = np.argsort(-s); match = (gpk[order] == qp[i])
        if not match.any():
            aps.append(0.0); r1s.append(0.0); false10.append(1.0); continue
        cum = np.cumsum(match); ranks = np.arange(1, len(match) + 1)
        aps.append((cum[match] / ranks[match]).mean()); r1s.append(float(match[0]))
        false10.append(float((gpk[order[:10]] != qp[i]).mean()))
    return 100 * np.mean(aps), 100 * np.mean(r1s), float(np.mean(false10))


def single_support_diag(qf, qp, qc, gf, gp, gc, seeds=20):
    id2g = defaultdict(list)
    for i, p in enumerate(gp): id2g[int(p)].append(i)
    q_ids = set(qp.tolist())
    hasq = [p for p in id2g if p in q_ids]
    distractor_g = np.asarray([i for p in id2g if p not in q_ids for i in id2g[p]], dtype=int)
    valid_q = np.asarray([i for i in range(len(qf))
                          if (gp[~((gp == qp[i]) & (gc == qc[i]))] == qp[i]).any()], dtype=int)

    def supp_g(sidx):
        chunks = [np.asarray(sidx, dtype=int)]
        if len(distractor_g): chunks.append(distractor_g)
        return np.concatenate(chunks)

    full_mAP, full_R1, full_f10 = eval_fixed(qf, qp, qc, gf, gp, gc, np.arange(len(gf)), valid_q)
    rand_mAPs, rand_f10s = [], []
    for sd in range(seeds):
        rng = np.random.RandomState(sd)
        r = eval_fixed(qf, qp, qc, gf, gp, gc, supp_g([rng.choice(id2g[p]) for p in hasq]), valid_q)
        rand_mAPs.append(r[0]); rand_f10s.append(r[2])
    best_s, worst_s = [], []
    for p in hasq:
        gi = id2g[p]; qs = np.where(qp == p)[0]
        qual = [(qf[qs[qc[qs] != gc[g]]] @ gf[g]).mean() if (qc[qs] != gc[g]).any() else -1 for g in gi]
        best_s.append(gi[int(np.argmax(qual))]); worst_s.append(gi[int(np.argmin(qual))])
    best_mAP, _, best_f10 = eval_fixed(qf, qp, qc, gf, gp, gc, supp_g(best_s), valid_q)
    worst_mAP, _, worst_f10 = eval_fixed(qf, qp, qc, gf, gp, gc, supp_g(worst_s), valid_q)
    print(f"  [DIAG] full={full_mAP:.2f}(R1 {full_R1:.2f} f10 {full_f10:.3f}) "
          f"best={best_mAP:.2f}(f10 {best_f10:.3f}) "
          f"random={np.mean(rand_mAPs):.2f}±{np.std(rand_mAPs):.2f}(f10 {np.mean(rand_f10s):.3f}±{np.std(rand_f10s):.3f}) "
          f"worst={worst_mAP:.2f}(f10 {worst_f10:.3f}) best-worst={best_mAP-worst_mAP:.2f}", flush=True)
    return full_mAP, np.mean(rand_mAPs), worst_mAP


def main():
    args = parse_args()
    set_seed(1234)
    cfg_setup(args)
    device = "cuda"
    train_loader, train_loader_normal, val_loader, num_query, num_classes, cam_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_classes, cam_num, view_num, cfg.MODEL.SEMANTIC_WEIGHT)
    model.load_param(args.ckpt)
    model = model.to(device)
    unfreeze_last_stage(model)
    K = cfg.DATALOADER.NUM_INSTANCE
    loss_func, center_criterion = make_loss(cfg, num_classes)
    optimizer, _ = make_optimizer(cfg, model, center_criterion)  # codex: 项目签名 (cfg,model,center)->(opt,opt_center)
    scheduler = create_scheduler(cfg, optimizer)
    use_amp = not args.no_amp
    scaler = amp.GradScaler(enabled=use_amp)

    for epoch in range(1, args.epochs + 1):
        model.train(); set_frozen_bn_eval(model)
        t0 = time.time(); loss_sum = acc_sum = n_sum = 0.0
        for it, batch in enumerate(train_loader):
            if cfg.MODEL.POSE_ENABLED:
                img, target, cam, view, pose_dict = batch
                pose_dict = _pose_to_device(pose_dict, device)
            else:
                img, target, cam, view = batch[:4]; pose_dict = None
            img = img.to(device); target = target.to(device); cam = cam.to(device); view = view.to(device)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(enabled=use_amp):
                # codex High: train 也强制 pose-off (pose_dict=None), 让 SS loss 和 eval descriptor 同口径(pose-off global), 避 LGPA/GCN list-loss 干扰
                out = model(img, label=target, cam_label=cam, view_label=view, pose_dict=None) \
                    if cfg.MODEL.POSE_ENABLED else model(img, label=target, cam_label=cam, view_label=view)
                score, feat, _, recon_loss, kp_data = unwrap_out(out)
                base_loss = call_project_loss(loss_func, score, feat, target, cam, kp_data)
                base_loss = add_aux_pose_losses(base_loss, cfg, kp_data, recon_loss)
                ss_ce = img.new_tensor(0.0); ss_cvar = img.new_tensor(0.0)
                if args.mode != "plain":
                    ss_feat = global_train_feat(feat)
                    ss_ce, ss_cvar = ss_episode_loss(ss_feat, target, K, args.alpha, args.tau)
                    ss_loss = ss_ce if args.mode == "random" else (ss_ce + args.lam * ss_cvar)
                    loss = base_loss + args.ss_weight * ss_loss
                else:
                    loss = base_loss
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            with torch.no_grad():
                s0 = score[0] if isinstance(score, (list, tuple)) else score
                acc = (s0.argmax(1) == target).float().mean().item()
            bs = img.size(0); loss_sum += float(loss.item()) * bs; acc_sum += acc * bs; n_sum += bs
            if it % cfg.SOLVER.LOG_PERIOD == 0:
                print(f"[e{epoch:03d} i{it:04d}/{len(train_loader)}] loss={loss_sum/n_sum:.4f} "
                      f"acc={acc_sum/n_sum:.3f} base={float(base_loss.item()):.4f} "
                      f"ss_ce={float(ss_ce.item()):.4f} ss_cvar={float(ss_cvar.item()):.4f}", flush=True)
        scheduler.step(epoch)  # codex: 项目 scheduler 要 step(epoch)
        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, f"transformer_laststage_{epoch}.pth"))
        print(f"[epoch {epoch}] done {time.time()-t0:.1f}s", flush=True)
        if args.eval_period > 0 and epoch % args.eval_period == 0:
            qf, qp, qc, gf, gp, gc = extract_global_eval(cfg, model, val_loader, num_query, device, args.flip_test)
            single_support_diag(qf, qp, qc, gf, gp, gc, seeds=args.eval_seeds)
    qf, qp, qc, gf, gp, gc = extract_global_eval(cfg, model, val_loader, num_query, device, args.flip_test)
    single_support_diag(qf, qp, qc, gf, gp, gc, seeds=args.eval_seeds)


if __name__ == "__main__":
    main()
