#!/usr/bin/env python3
"""exp364 DG bounded retest：SOLIDER swin person-pretrain frozen base probe。

codex 判 DINOv2 frozen ReID 弱（Market 2.71）→ DINOv2 frozen prior preservation 死。
bounded retest 首选 SOLIDER（person-domain pretrain swin，pretrained/swin_tiny.pth，非 Market fine-tuned）。
判定线（codex）：frozen base Market in-domain < 10 mAP → 不许作 topology preservation teacher；
SOLIDER person-pretrain 也弱 → DG foundation-preserving 降级转 open-set/gallery-growth。

复用 frozen_xdomain_probe 的 market_list/eval_reid，只换 encoder 为项目 SOLIDER swin（PSC-JEPA build_backbone 方式）。
"""
import os, sys, glob, re, argparse, numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

IMG_H, IMG_W = 256, 128


def build_encoder(weights, device, repo='.', semantic_weight=0.2, ckpt=None):
    sys.path.insert(0, repo)
    from model.backbones.swin_transformer import swin_tiny_patch4_window7_224
    net = swin_tiny_patch4_window7_224(img_size=[IMG_H, IMG_W], drop_path_rate=0.1, drop_rate=0.0,
                                       attn_drop_rate=0.0, pretrained=weights,
                                       convert_weights=False, semantic_weight=semantic_weight)
    net.init_weights(weights)                             # SOLIDER person-pretrain
    if ckpt:                                              # 训练 ckpt 覆盖 backbone（strip base. prefix, codex 审）
        sd = torch.load(ckpt, map_location='cpu')
        sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
        bk = {k[5:]: v for k, v in sd.items() if k.startswith('base.')}
        if bk:
            net.load_state_dict(bk, strict=False)
        print(f"[ckpt] loaded {len(bk)} base. keys from {ckpt}", flush=True)
    net = net.to(device)
    net.eval()                                            # SOLIDER swin .eval() 不返回 self（PSC-JEPA 踩过），不能 chain
    tf = T.Compose([T.Resize((IMG_H, IMG_W)), T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # match SOLIDER 训练 config [0.5]*3（codex: frozen/FT 同口径）
    return tf, net


@torch.no_grad()
def encode(net, tf, paths, device, bs=128):
    feats = []
    for i in range(0, len(paths), bs):
        imgs = torch.stack([tf(Image.open(p).convert('RGB')) for p in paths[i:i + bs]]).to(device)
        out = net(imgs)
        gf = out[0] if isinstance(out, (tuple, list)) else out   # SOLIDER swin out[0] = GAP global feat
        feats.append(F.normalize(gf, dim=1).cpu())
    return torch.cat(feats)


def market_list(d):
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
    items = []
    with open(listfile) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            rel, pid = parts[0], parts[1]
            # MSMT 文件名 pid_xxx_camid_...（如 0000_000_01_...），camid 是第 3 段（无 _c 前缀；
            # 之前 re _c 解析失败→camid 全 0→eval 把同 pid 正样本 exclude 光→nan）
            cam = int(os.path.basename(rel).split('_')[2])
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


def run_domain(name, q_items, g_items, net, tf, device):
    if not q_items or not g_items:
        print(f"[{name}] SKIP (q={len(q_items)} g={len(g_items)})")
        return
    qp = np.array([x[1] for x in q_items]); qc = np.array([x[2] for x in q_items])
    gp = np.array([x[1] for x in g_items]); gc = np.array([x[2] for x in g_items])
    qf = encode(net, tf, [x[0] for x in q_items], device)
    gf = encode(net, tf, [x[0] for x in g_items], device)
    mAP, r1 = eval_reid(qf, qp, qc, gf, gp, gc)
    print(f"[{name}] frozen SOLIDER-swin-tiny  q={len(q_items)} g={len(g_items)}  mAP={mAP:.2f}  R1={r1:.2f}"
          f"  [dim={qf.shape[1]}]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='pretrained/swin_tiny.pth')
    ap.add_argument('--ckpt', default=None)               # 训练 ckpt（direct-FT/head-only）覆盖 backbone
    ap.add_argument('--dataroot', default='data')
    ap.add_argument('--repo', default='.')
    ap.add_argument('--device', default='cuda')
    cli = ap.parse_args()
    tf, net = build_encoder(cli.weights, cli.device, cli.repo, ckpt=cli.ckpt)
    print(f"[exp364 SOLIDER frozen probe] swin_tiny person-pretrain frozen, in-domain mAP baseline ({IMG_H}x{IMG_W})")

    dr = cli.dataroot
    run_domain('Market', market_list(f'{dr}/market1501/query'),
               market_list(f'{dr}/market1501/bounding_box_test'), net, tf, cli.device)
    run_domain('Occ-Duke', market_list(f'{dr}/occluded_duke/query'),
               market_list(f'{dr}/occluded_duke/bounding_box_test'), net, tf, cli.device)
    try:
        run_domain('MSMT17', msmt_list(f'{dr}/MSMT17/list_query.txt', f'{dr}/MSMT17/test'),
                   msmt_list(f'{dr}/MSMT17/list_gallery.txt', f'{dr}/MSMT17/test'), net, tf, cli.device)
    except Exception as e:
        print(f"[MSMT17 skip] {e}")
    print("[done]")


if __name__ == '__main__':
    main()
