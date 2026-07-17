#!/usr/bin/env python3
"""exp364 DG cheap kill-switch 第一步：frozen DINOv2-reg cross-domain probe（零训练）。

验证 frozen foundation 的 ReID 基线：DINOv2-reg frozen 提特征，Market/Occ-Duke/MSMT in-domain mAP/R1。
这是 fine-tune 对比的起点——fine-tune 要超 frozen 才有意义，破坏 frozen 则 fine-tune-harm。
后续第二步 head-only/direct-FT 对比，看 U-shaped sweet spot（preservation 是否有救还是 no-op）。
"""
import os, sys, glob, re, argparse, numpy as np, torch
import torch.nn.functional as F
from PIL import Image
import timm
from torchvision import transforms as T


def build_encoder(weights, device):
    model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False,
                              num_classes=0, dynamic_img_size=True)
    model.load_state_dict(torch.load(weights, map_location='cpu'), strict=True)
    model = model.to(device).eval()
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),   # DINOv2 方形原生 16x16 patch14（252x126 非方形 dynamic 退化 debug）
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return tf, model


@torch.no_grad()
def encode(model, tf, paths, device, bs=128):
    feats = []
    for i in range(0, len(paths), bs):
        imgs = torch.stack([tf(Image.open(p).convert('RGB')) for p in paths[i:i + bs]]).to(device)
        feat = model.forward_features(imgs)        # [B, 1 CLS + 4 register + N patch, D]
        gap = feat[:, 5:].mean(1)                  # patch tokens GAP（skip CLS+4register；CLS 太 semantic，instance ReID 用 patch GAP）
        feats.append(F.normalize(gap, dim=1).cpu())
    return torch.cat(feats)


def market_list(d):
    """market/duke 格式：pid_cXsX_...jpg。pid<0 是 junk 跳过。"""
    items = []
    for p in sorted(glob.glob(os.path.join(d, '*.jpg'))):
        m = re.match(r'(-?\d+)_c(\d+)', os.path.basename(p))
        if not m:
            continue
        pid, cam = int(m.group(1)), int(m.group(2))
        if pid < 0:
            continue
        items.append((p, pid, cam))
    return items


def msmt_list(listfile, root):
    """MSMT: 每行 'rel_path pid'，camid 从文件名 xxxx_cYY_ 取。"""
    items = []
    with open(listfile) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            rel, pid = parts[0], parts[1]
            mm = re.search(r'_c(\d+)', os.path.basename(rel))
            cam = int(mm.group(1)) if mm else 0
            items.append((os.path.join(root, rel), int(pid), cam))
    return items


def eval_reid(qf, qp, qc, gf, gp, gc):
    dist = (1 - qf @ gf.t()).numpy()
    aps, r1 = [], []
    for i in range(len(qp)):
        o = np.argsort(dist[i])
        keep = ~((gp[o] == qp[i]) & (gc[o] == qc[i]))
        oo = o[keep]
        m = (gp[oo] == qp[i]).astype(np.int32)
        if m.sum() == 0:
            continue
        r1.append(m[0])
        cs = m.cumsum()
        prec = cs / (np.arange(len(m)) + 1)
        aps.append((prec * m).sum() / m.sum())
    return float(np.mean(aps)) * 100, float(np.mean(r1)) * 100


def run_domain(name, q_items, g_items, model, tf, device):
    if not q_items or not g_items:
        print(f"[{name}] SKIP (empty: q={len(q_items)} g={len(g_items)})")
        return
    qp = np.array([x[1] for x in q_items]); qc = np.array([x[2] for x in q_items])
    gp = np.array([x[1] for x in g_items]); gc = np.array([x[2] for x in g_items])
    qf = encode(model, tf, [x[0] for x in q_items], device)
    gf = encode(model, tf, [x[0] for x in g_items], device)
    mAP, r1 = eval_reid(qf, qp, qc, gf, gp, gc)
    print(f"[{name}] frozen DINOv2-reg  q={len(q_items)} g={len(g_items)}  mAP={mAP:.2f}  R1={r1:.2f}"
          f"  [qcam={list(np.unique(qc))[:6]} gcam={list(np.unique(gc))[:6]} dim={qf.shape[1]} "
          f"qpid={len(np.unique(qp))} gpid={len(np.unique(gp))}]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='experiments/exp363_ag_foundation/dinov2reg_timm.pth')
    ap.add_argument('--dataroot', default='data')
    ap.add_argument('--device', default='cuda')
    cli = ap.parse_args()
    tf, model = build_encoder(cli.weights, cli.device)
    print("[exp364 frozen cross-domain probe] DINOv2-reg frozen, in-domain mAP baseline (256x128)")

    dr = cli.dataroot
    run_domain('Market', market_list(f'{dr}/market1501/query'),
               market_list(f'{dr}/market1501/bounding_box_test'), model, tf, cli.device)
    run_domain('Occ-Duke', market_list(f'{dr}/occluded_duke/query'),
               market_list(f'{dr}/occluded_duke/bounding_box_test'), model, tf, cli.device)
    try:
        run_domain('MSMT17', msmt_list(f'{dr}/MSMT17/list_query.txt', f'{dr}/MSMT17/test'),
                   msmt_list(f'{dr}/MSMT17/list_gallery.txt', f'{dr}/MSMT17/test'), model, tf, cli.device)
    except Exception as e:
        print(f"[MSMT17 skip] {e}")
    print("[done]")


if __name__ == '__main__':
    main()
