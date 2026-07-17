#!/usr/bin/env python3
"""exp363 AG-VPReID.VIR frozen DINOv2-reg-B baseline — cheap kill-switch 第一步（无训练）。

codex 硬判定（任一不过即杀，不补 LoRA/attention 小变体）：
- frozen temporal mean vs single frame：hard bucket ≥ +5 mAP/R1（视频证据积累成立）
- oracle vs mean ≥ +3 mAP（选择/校准有空间，留给 anchored-LoRA）

逻辑：dataset_reader.AG_ReID_IR_Enhanced 读 query/gallery tracklet → DINOv2-reg-B frozen CLS encode
每帧 → temporal pooling（single/mean/max/topk/oracle）→ tracklet feat → mAP/R1 per-protocol。
先验证 frozen foundation 在 AG 上的视频/模态 headroom，再决定是否上 anchored-LoRA。
"""
import os, sys, argparse, numpy as np, torch
import torch.nn.functional as F
from PIL import Image
import timm
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # 让 dataset_reader.py 可 import（rsync 时一起拷到 exp363 目录）


def build_encoder(weights, device):
    # timm DINOv2-reg（transformers 4.46 不支持 dinov2_with_registers，3090 python3.8 装不了新版）
    model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False,
                              num_classes=0, dynamic_img_size=True)
    sd = torch.load(weights, map_location='cpu')
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    return tf, model


@torch.no_grad()
def encode_frames(model, tf, img_paths, device, bs=64):
    """每帧 DINOv2-reg-B pooled CLS → L2 norm。返回 [Nframes, D]。"""
    feats = []
    for i in range(0, len(img_paths), bs):
        imgs = torch.stack([tf(Image.open(p).convert('RGB')) for p in img_paths[i:i + bs]]).to(device)
        out = model(imgs)                                     # num_classes=0 → pooled CLS [b, D]
        feats.append(F.normalize(out, dim=1).cpu())
    return torch.cat(feats) if feats else torch.zeros(0)


def pool(fr, mode, k=4):
    """fr [N,D] L2-normed → tracklet feat [D]（再 L2 norm）。"""
    if fr.shape[0] == 0:
        return None
    if mode == 'single':
        v = fr[0]                                             # 第一帧
    elif mode == 'mean':
        v = fr.mean(0)
    elif mode == 'max':
        v = fr.max(0)[0]
    elif mode == 'topk':                                     # 离 tracklet 中心最近的 k 帧（quality 代理）
        c = F.normalize(fr.mean(0, keepdim=True), dim=1)
        sim = (fr * c).sum(1)
        idx = sim.topk(min(k, fr.shape[0]))[1]
        v = fr[idx].mean(0)
    else:
        raise ValueError(mode)
    return F.normalize(v, dim=0)


def encode_tracklets(tracklets, model, proc, device, nframes, mode):
    """tracklets: list of (img_paths, pid, camid)。返回 feats[M,D], pids[M], cams[M]。"""
    feats, pids, cams = [], [], []
    for imgs, pid, cam in tracklets:
        sel = imgs if nframes <= 0 or len(imgs) <= nframes else \
            [imgs[j] for j in np.linspace(0, len(imgs) - 1, nframes).astype(int)]
        fr = encode_frames(model, proc, sel, device)
        v = pool(fr, mode)
        if v is None:
            continue
        feats.append(v); pids.append(pid); cams.append(cam)
    return torch.stack(feats), np.array(pids), np.array(cams)


def eval_map_r1(qf, qp, qc, gf, gp, gc):
    """标准 ReID：cosine 距离，排除同 pid&同 cam 的 gallery。返回 mAP, R1。"""
    dist = 1 - qf @ gf.t()                                   # [Q,G]，feat 已 L2 norm
    dist = dist.numpy()
    aps, r1s = [], []
    for i in range(len(qp)):
        order = np.argsort(dist[i])
        keep = ~((gp[order] == qp[i]) & (gc[order] == qc[i]))
        o = order[keep]
        matches = (gp[o] == qp[i]).astype(np.int32)
        if matches.sum() == 0:
            continue
        r1s.append(matches[0])
        cs = matches.cumsum()
        prec = cs / (np.arange(len(matches)) + 1)
        aps.append((prec * matches).sum() / matches.sum())
    return float(np.mean(aps)) * 100, float(np.mean(r1s)) * 100


def oracle_map_r1(q_fr_list, qp, qc, gf, gp, gc):
    """oracle pooling upper bound：每 query tracklet 选使 AP 最大的单帧（用真 label）。"""
    aps, r1s = [], []
    for i, fr in enumerate(q_fr_list):
        if fr.shape[0] == 0:
            continue
        best_ap, best_r1 = 0.0, 0
        for f in fr:                                         # 试每一帧
            d = (1 - (f.unsqueeze(0) @ gf.t())).numpy()[0]
            order = np.argsort(d)
            keep = ~((gp[order] == qp[i]) & (gc[order] == qc[i]))
            o = order[keep]
            m = (gp[o] == qp[i]).astype(np.int32)
            if m.sum() == 0:
                continue
            cs = m.cumsum(); prec = cs / (np.arange(len(m)) + 1)
            ap = (prec * m).sum() / m.sum()
            if ap > best_ap:
                best_ap, best_r1 = ap, m[0]
        aps.append(best_ap); r1s.append(best_r1)
    return float(np.mean(aps)) * 100, float(np.mean(r1s)) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/root/work/SOLIDER-REID/data/ag_vir')
    ap.add_argument('--weights', default='/root/work/SOLIDER-REID/dinov2reg')
    ap.add_argument('--exp', default='exp7')                 # cross_platform A-G visible_infrared
    ap.add_argument('--nframes', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    cli = ap.parse_args()

    sys.path.insert(0, cli.root)                            # dataset_reader.py 在数据根目录 ag_vir/
    from dataset_reader import AG_ReID_IR_Enhanced
    ds = AG_ReID_IR_Enhanced(root_path=cli.root, experiment=cli.exp, use_organized_structure=True)
    proc, model = build_encoder(cli.weights, cli.device)
    print(f"[exp363 frozen baseline] exp={cli.exp} nframes={cli.nframes} weights={cli.weights}")

    # encode gallery once (mean pooling 固定，公平对比 query pooling)
    gf, gp, gc = encode_tracklets(ds.gallery, model, proc, cli.device, cli.nframes, 'mean')
    print(f"gallery: {len(gp)} tracklets")

    rows = []
    for mode in ['single', 'mean', 'max', 'topk']:
        qf, qp, qc = encode_tracklets(ds.query, model, proc, cli.device, cli.nframes, mode)
        mAP, r1 = eval_map_r1(qf, qp, qc, gf, gp, gc)
        rows.append((mode, mAP, r1))
        print(f"  {mode:8s}  mAP={mAP:5.2f}  R1={r1:5.2f}")

    # oracle（per-query 选最佳帧，需逐帧 feat）
    q_fr = [encode_frames(model, proc,
            (im if cli.nframes <= 0 or len(im) <= cli.nframes else
             [im[j] for j in np.linspace(0, len(im) - 1, cli.nframes).astype(int)]),
            cli.device) for im, _, _ in ds.query]
    qp = np.array([p for _, p, _ in ds.query]); qc = np.array([c for _, _, c in ds.query])
    oap, or1 = oracle_map_r1(q_fr, qp, qc, gf, gp, gc)
    print(f"  {'oracle':8s}  mAP={oap:5.2f}  R1={or1:5.2f}")

    # codex 硬判定
    d = dict((m, v) for m, v, _ in rows)
    print("\n[codex 硬判定]")
    print(f"  mean - single = {d['mean']-d['single']:+.2f} mAP  (需 hard bucket +5 → 视频证据积累成立)")
    print(f"  oracle - mean = {oap-d['mean']:+.2f} mAP  (需 +3 → 选择/校准空间留给 anchored-LoRA)")
    print("[done]")


if __name__ == '__main__':
    main()
