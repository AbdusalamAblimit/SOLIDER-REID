#!/usr/bin/env python3
"""exp361 PSC-JEPA continued-pretrain — Stage-A skeleton (same-image EMA teacher, no support bank yet).

Idea: continued-pretrain SOLIDER swin_tiny so the backbone learns "how to form retrievable identity
evidence when a body part is missing". JEPA in LATENT body-part-token space (not pixel):
  teacher (EMA, sees FULL image) -> part tokens T[B,G,C]
  student (sees pose-defined PARTIAL view, some body groups masked) -> part tokens S[B,G,C]
  L_part_jepa: student must PREDICT the dropped-group teacher tokens from the visible context.
  L_visible_anchor: visible-group student tokens stay close to teacher (don't drift).
  L_union: pooled identity token (student) close to teacher full identity token.

Stage-A intentionally uses ONLY same-image EMA teacher (= codex's 3090 control "is it just OA-SD/PCVT
renamed"). The MAIN novelty (pseudo same-ID support bank T_bank) is Stage-B, added on top once this
skeleton trains stably. Coordinate-system / loss-collapse points are flagged @REVIEW for codex审 diff.

Run: python psc_jepa_pretrain.py --backbone pretrained/swin_tiny.pth --data data/occluded_duke \
        --pose data/occluded_duke/pose_train.npz --epochs 50 --out log/.../exp361_pscjepa_A
"""
import sys, os, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
ap = argparse.ArgumentParser()
ap.add_argument('--repo', default='/home/afr/SOLIDER-REID')
ap.add_argument('--backbone', default='pretrained/swin_tiny.pth')
ap.add_argument('--data', default='data/occluded_duke')          # has bounding_box_train/
ap.add_argument('--pose', default='data/occluded_duke/pose_train.npz')
ap.add_argument('--epochs', type=int, default=50)
ap.add_argument('--bs', type=int, default=64)
ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--ema', type=float, default=0.996)
ap.add_argument('--drop_groups', type=int, default=2)            # how many visible groups to mask for student
ap.add_argument('--w_jepa', type=float, default=1.0)
ap.add_argument('--w_anchor', type=float, default=1.0)
ap.add_argument('--w_union', type=float, default=0.5)
ap.add_argument('--vis_thr', type=float, default=0.3)            # keypoint visibility threshold
ap.add_argument('--semantic_weight', type=float, default=0.2)    # H2: match downstream swin_tiny (was hardcoded 1.0)
ap.add_argument('--w_var', type=float, default=1.0)              # C1: variance regularization weight (anti-collapse)
ap.add_argument('--w_solider_anchor', type=float, default=1.0)   # Stage-B 防遗忘: visible-token 蒸 frozen SOLIDER (0=Stage-A control, 已证有害)
ap.add_argument('--anchor_path', default='pretrained/swin_tiny.pth')  # frozen SOLIDER anchor (原始, 非 continued-pretrain)
ap.add_argument('--w_global_anchor', type=float, default=1.0)        # Stage-B v2 (codex R1): global GAP distill 权重 (锚全局判别几何, 防 forgetting 核心)
ap.add_argument('--save_every', type=int, default=10)
ap.add_argument('--out', default='log/occluded_duke/exp361_pscjepa_A')
ap.add_argument('--smoke', type=int, default=0)                  # smoke: cap #images
cli = ap.parse_args()
os.chdir(cli.repo); sys.path.insert(0, cli.repo)
DEV = 'cuda'
IMG_H, IMG_W = 384, 128
GH, GW = 12, 4                                                   # swin /32 featmap grid for 384x128

# --- COCO-17 keypoints -> 5 body groups (indices) ---
# 0 nose,1-2 eyes,3-4 ears,5-6 shoulders,7-8 elbows,9-10 wrists,11-12 hips,13-14 knees,15-16 ankles
GROUPS = {
    'head':  [0, 1, 2, 3, 4],
    'torso': [5, 6, 11, 12],
    'larm':  [5, 7, 9],
    'rarm':  [6, 8, 10],
    'legs':  [11, 12, 13, 14, 15, 16],
}
GKEYS = list(GROUPS.keys()); G = len(GKEYS)


def load_pose(npz):
    d = np.load(os.path.join(cli.repo, npz), allow_pickle=True)
    fn = d['filenames']; vis = d['visibility'].astype(np.float32); kp = d['keypoints'].astype(np.float32)
    return {fn[i]: (kp[i], vis[i]) for i in range(len(fn))}                  # name -> (kp[17,2], vis[17])


class PSCDataset(torch.utils.data.Dataset):
    """image (384x128 tensor) + per-group visibility mask + per-group featmap-grid region mask."""
    def __init__(self, data, pose):
        self.dir = os.path.join(cli.repo, data, 'bounding_box_train')
        self.pose = load_pose(pose)
        self.items = [f for f in sorted(os.listdir(self.dir)) if f.endswith('.jpg') and f in self.pose]
        if cli.smoke:
            self.items = self.items[:cli.smoke]
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.items)

    def _grid_region(self, kp, vis):
        """Per-group: visibility (any kp in group visible) + bbox region mask on GHxGW grid.
        @REVIEW codex: keypoints are pixel coords in the ORIGINAL crop; we read crop size and normalize
        to [0,1] then to grid. If pose npz coords are already resized to 384x128, drop the /w,/h."""
        gvis = np.zeros(G, np.float32); gmask = np.zeros((G, GH, GW), np.float32)
        for gi, gk in enumerate(GKEYS):
            idx = GROUPS[gk]
            v = vis[idx]; pts = kp[idx][v >= cli.vis_thr]
            if len(pts) == 0:
                continue
            gvis[gi] = 1.0
            xs = np.clip((pts[:, 0]) * GW, 0, GW - 1e-3); ys = np.clip((pts[:, 1]) * GH, 0, GH - 1e-3)
            x0, x1 = int(xs.min()), int(xs.max()); y0, y1 = int(ys.min()), int(ys.max())
            gmask[gi, y0:y1 + 1, x0:x1 + 1] = 1.0
        return gvis, gmask

    def __getitem__(self, i):
        name = self.items[i]
        im = Image.open(os.path.join(self.dir, name)).convert('RGB')
        ow, oh = im.size
        im = im.resize((IMG_W, IMG_H))
        t = torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0)
        t = (t - self.mean) / self.std
        kp, vis = self.pose[name]
        kpn = kp.copy(); kpn[:, 0] /= max(ow, 1); kpn[:, 1] /= max(oh, 1)    # normalize to [0,1]
        gvis, gmask = self._grid_region(kpn, vis)
        return t, torch.from_numpy(gvis), torch.from_numpy(gmask)


def build_backbone(path=None):
    path = path or cli.backbone
    from model.backbones.swin_transformer import swin_tiny_patch4_window7_224
    net = swin_tiny_patch4_window7_224(img_size=[IMG_H, IMG_W], drop_path_rate=0.1, drop_rate=0.0,
                                       attn_drop_rate=0.0, pretrained=os.path.join(cli.repo, path),
                                       convert_weights=False, semantic_weight=cli.semantic_weight)  # H2: match downstream 0.2
    net.init_weights(os.path.join(cli.repo, path))
    return net


def part_pool(featmaps, gmask):
    """featmaps[B,C,Hf,Wf] + gmask[B,G,Hf,Wf] -> part tokens[B,G,C] (masked avg-pool, L2)."""
    B, C, Hf, Wf = featmaps.shape
    if (Hf, Wf) != (GH, GW):
        gmask = F.interpolate(gmask, size=(Hf, Wf), mode='nearest')
    denom = gmask.sum((2, 3)).clamp_min(1.0)                                 # [B,G]
    tok = torch.einsum('bchw,bghw->bgc', featmaps, gmask) / denom.unsqueeze(-1)
    return F.normalize(tok, dim=2)


def fwd_tokens(net, x, gmask):
    out = net(x)
    global_feat = out[0] if isinstance(out, (tuple, list)) else None         # GAP global feat (out[0])
    featmaps = out[1] if isinstance(out, (tuple, list)) else out
    if isinstance(featmaps, (tuple, list)):
        featmaps = featmaps[-1]
    gf = F.normalize(global_feat, dim=1) if global_feat is not None else None
    return part_pool(featmaps, gmask), gf                                    # (part [B,G,C], global [B,D] L2)


def main():
    ds = PSCDataset(cli.data, cli.pose)
    bs = min(cli.bs, len(ds)) if cli.smoke else cli.bs                      # v2-Med: smoke<bs 不静默空跑
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=8, drop_last=(not cli.smoke))
    print(f"[PSC-JEPA] {len(ds)} train imgs, {len(dl)} iters/ep, G={G} groups", flush=True)
    student = build_backbone().to(DEV); student.train()                    # SOLIDER swin .train()/.eval() 不返回 self, 不能链式调用
    teacher = build_backbone().to(DEV); teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.load_state_dict(student.state_dict())
    anchor = None                                                          # Stage-B 防遗忘: frozen SOLIDER anchor (原始 swin_tiny, 不更新)
    if cli.w_solider_anchor > 0:
        anchor = build_backbone(cli.anchor_path).to(DEV); anchor.eval()
        for p in anchor.parameters():
            p.requires_grad_(False)
    C = student.num_features[-1]                                            # C1: asymmetric predictor (BYOL/data2vec anti-collapse)
    predictor = nn.Sequential(nn.Linear(C, C), nn.BatchNorm1d(C), nn.GELU(), nn.Linear(C, C)).to(DEV)
    opt = torch.optim.AdamW(list(student.parameters()) + list(predictor.parameters()), lr=cli.lr, weight_decay=0.05)
    os.makedirs(os.path.join(cli.repo, cli.out), exist_ok=True)
    for ep in range(cli.epochs):
        last = {}
        for it, (x, gvis, gmask) in enumerate(dl):
            x, gvis, gmask = x.to(DEV), gvis.to(DEV), gmask.to(DEV)          # [B,3,H,W],[B,G],[B,G,GH,GW]
            B = x.shape[0]
            if B < 2:                                                       # Low: predictor BatchNorm1d needs B>=2 (smoke 尾批保护)
                continue
            # teacher sees FULL image
            with torch.no_grad():
                T, _ = fwd_tokens(teacher, x, gmask)                         # [B,G,C]
            # student sees PARTIAL: drop `drop_groups` visible groups by masking image region
            drop = torch.zeros(B, G, device=DEV)
            for b in range(B):
                vis_g = torch.where(gvis[b] > 0)[0]
                nd = min(cli.drop_groups, max(0, len(vis_g) - 1))           # M1: keep >=1 visible, 不排除重遮挡样本(目标人群)
                if nd > 0:
                    sel = vis_g[torch.randperm(len(vis_g), device=DEV)[:nd]]
                    drop[b, sel] = 1.0
            # build image-space drop mask from grid region of dropped groups
            dmask_grid = (gmask * drop.unsqueeze(-1).unsqueeze(-1)).sum(1).clamp(0, 1)   # [B,GH,GW]
            dmask_img = F.interpolate(dmask_grid.unsqueeze(1), size=(IMG_H, IMG_W), mode='nearest')
            x_part = x * (1.0 - dmask_img)                                   # zero-out dropped region (post-norm≈mean, ok as occlusion)
            S, _ = fwd_tokens(student, x_part, gmask)                        # [B,G,C]
            Sp = F.normalize(predictor(S.reshape(-1, C)).reshape(B, G, C), dim=2)  # C1: predictor(student) predicts teacher
            # losses (cosine in latent token space, predictor-asymmetric)
            cos = (Sp * T).sum(2)                                          # [B,G] predicted-vs-teacher sim
            drop_m = drop * gvis                                           # only dropped & originally-visible
            vis_m = (1.0 - drop) * gvis                                     # visible & kept
            L_jepa = ((1.0 - cos) * drop_m).sum() / drop_m.sum().clamp_min(1.0)   # predict dropped from context
            L_anchor = ((1.0 - cos) * vis_m).sum() / vis_m.sum().clamp_min(1.0)
            su = F.normalize(predictor((S * gvis.unsqueeze(-1)).sum(1)), dim=1)   # union identity token (predicted)
            tu = F.normalize((T * gvis.unsqueeze(-1)).sum(1), dim=1)
            L_union = (1.0 - (su * tu).sum(1)).mean()
            vmask = gvis.reshape(-1).bool()                                 # v2-High: only visible-group tokens (drop zero-vectors)
            Svis = S.reshape(-1, C)[vmask]
            std_s = (Svis.std(0) * (C ** 0.5)) if Svis.shape[0] > 1 else torch.ones(C, device=DEV)  # ×√C: healthy≈1, collapse→0
            L_var = F.relu(1.0 - std_s).mean()                              # target 1 now reachable
            L_anc_sol = torch.zeros((), device=DEV)                         # Stage-B 防遗忘: visible part token + global GAP 蒸 frozen SOLIDER
            sol_p, sol_g = 0.0, 0.0                                         # codex R2: 拆 part/global anchor 分项监控
            if anchor is not None:
                with torch.no_grad():
                    A, A_gf = fwd_tokens(anchor, x, gmask)                 # frozen SOLIDER full-view part [B,G,C] + global [B,D]
                _, S_gf_full = fwd_tokens(student, x, gmask)              # student FULL-view global (有grad, 锚全局判别几何 codex-R1)
                L_part = ((1.0 - (S * A).sum(2)) * vis_m).sum() / vis_m.sum().clamp_min(1.0)  # 局部可见区
                L_glob = (1.0 - (S_gf_full * A_gf).sum(1)).mean()        # ★全局 GAP 判别几何 (防 forgetting 核心 codex-R1)
                L_anc_sol = L_part + cli.w_global_anchor * L_glob
                sol_p, sol_g = L_part.item(), L_glob.item()
            loss = (cli.w_jepa * L_jepa + cli.w_anchor * L_anchor + cli.w_union * L_union
                    + cli.w_var * L_var + cli.w_solider_anchor * L_anc_sol)
            opt.zero_grad(); loss.backward(); opt.step()
            # EMA teacher update
            with torch.no_grad():
                for ps, pt in zip(student.parameters(), teacher.parameters()):
                    pt.mul_(cli.ema).add_(ps.detach(), alpha=1 - cli.ema)
            last = dict(L=loss.item(), jepa=L_jepa.item(), anchor=L_anchor.item(), union=L_union.item(),
                        var=L_var.item(), sol_p=sol_p, sol_g=sol_g, tok_std=std_s.mean().item(),  # codex R2: 拆 part/global anchor 分项
                        cos_drop=(cos * drop_m).sum().item() / drop_m.sum().clamp_min(1).item())
            if it % 50 == 0:
                print(f"  ep{ep} it{it}/{len(dl)} L={last['L']:.4f} jepa={last['jepa']:.4f} anchor={last['anchor']:.4f} "
                      f"union={last['union']:.4f} var={last['var']:.4f} solP={last['sol_p']:.4f} solG={last['sol_g']:.4f} tokStd={last['tok_std']:.3f} cosDrop={last['cos_drop']:.3f}", flush=True)
        print(f"[epoch {ep}] " + " ".join(f"{k}={v:.4f}" for k, v in last.items()), flush=True)
        if ((ep + 1) % cli.save_every == 0 or ep == cli.epochs - 1) and last:  # Low(r2): skip save if 0 update (empty/degenerate batch)
            sd = {f"backbone.{k}": v for k, v in student.state_dict().items()}  # H1: backbone. prefix for downstream init_weights
            torch.save({'state_dict': sd}, os.path.join(cli.repo, cli.out, f"pscjepa_{ep+1}.pth"))
            print(f"[save] pscjepa_{ep+1}.pth ({len(sd)} keys, backbone. prefix)", flush=True)
    print("[done] PSC-JEPA stage-A pretrain complete.")


if __name__ == '__main__':
    main()
