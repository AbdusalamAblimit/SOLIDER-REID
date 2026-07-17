# AFD-ReID kill-switch: 航拍↔地面检索中, 高频带是否比低/中频显著更不可靠?
# 无训练: pretrained resnet50 抽特征, 对 原图/low-pass/high-pass 各算 A<->G mAP。
# PASS: high << low/mid (或 full < gated-oracle) → confound 成立。FAIL: 各带同涨同跌 → 判死。
import torch, torchvision, glob, os, re, random
import numpy as np
from PIL import Image
import torchvision.transforms as T

device = 'cuda'
m = torchvision.models.resnet50(weights='IMAGENET1K_V1'); m.fc = torch.nn.Identity(); m = m.to(device).eval()  # V1=cached resnet50-19c8e357
tf = T.Compose([T.Resize((256,128)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def fft_band(x, mode, r=0.125):  # x:(3,H,W) tensor
    H,W = x.shape[-2:]
    f = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2,-1))
    cy,cx = H//2, W//2; ry,rx = max(1,int(H*r)), max(1,int(W*r))
    mask = torch.zeros((H,W))
    mask[cy-ry:cy+ry, cx-rx:cx+rx] = 1
    if mode == 'low':  f = f*mask
    elif mode == 'high': f = f*(1-mask)
    elif mode == 'mid':  # ring: keep 0.125~0.30, drop innermost + outer
        m2 = torch.zeros((H,W)); ry2,rx2 = int(H*0.30),int(W*0.30)
        m2[cy-ry2:cy+ry2, cx-rx2:cx+rx2] = 1; f = f*(m2-mask)
    return torch.fft.ifft2(torch.fft.ifftshift(f, dim=(-2,-1))).real

def extract(paths, mode):
    feats = []
    for i in range(0, len(paths), 128):
        imgs = []
        for p in paths[i:i+128]:
            img = tf(Image.open(p).convert('RGB'))
            if mode != 'orig': img = fft_band(img, mode)
            imgs.append(img)
        with torch.no_grad():
            feats.append(m(torch.stack(imgs).to(device)).cpu())
    return torch.cat(feats)

def cmap(qf, qp, gf, gp):
    qf = torch.nn.functional.normalize(qf); gf = torch.nn.functional.normalize(gf)
    sim = (qf @ gf.t()).numpy(); aps = []; r1 = 0; n = 0
    for i in range(len(qp)):
        o = sim[i].argsort()[::-1]; mt = (gp[o] == qp[i])
        if mt.sum() == 0: continue
        n += 1; r1 += mt[0]; c = np.cumsum(mt); aps.append((c/(np.arange(len(mt))+1)*mt).sum()/mt.sum())
    return (np.mean(aps)*100 if aps else 0), (r1/n*100 if n else 0)

imgs = glob.glob('/home/afr/SOLIDER-REID/data/**/Cam*/*.jpg', recursive=True)
print(f"found {len(imgs)} CARGO images")
A, G = [], []
for p in imgs:
    mt = re.search(r'[Cc]am(\d+)', p)
    if not mt: continue
    cam = int(mt.group(1)); name = os.path.basename(p)
    try: pid = int(name.split('_')[2])
    except: continue
    (A if cam <= 5 else G).append((p, pid))
print(f"Aerial(cam1-5)={len(A)} Ground(cam6-13)={len(G)}")
random.seed(0)
A = random.sample(A, min(3000, len(A))); G = random.sample(G, min(3000, len(G)))
Ap = np.array([x[1] for x in A]); Gp = np.array([x[1] for x in G])
Apaths = [x[0] for x in A]; Gpaths = [x[0] for x in G]
print(f"sampled A={len(A)} G={len(G)}, A∩G pids={len(set(Ap)&set(Gp))}")
print("="*60)
res = {}
for mode in ['orig','low','mid','high']:
    Af = extract(Apaths, mode); Gf = extract(Gpaths, mode)
    m1,r1 = cmap(Af, Ap, Gf, Gp); m2,r2 = cmap(Gf, Gp, Af, Ap)
    res[mode] = (m1,m2)
    print(f"[{mode:4s}] A->G mAP={m1:5.2f} R1={r1:5.2f} | G->A mAP={m2:5.2f} R1={r2:5.2f}")
print("="*60)
print(f"判定: orig={res['orig'][0]:.1f} low={res['low'][0]:.1f} mid={res['mid'][0]:.1f} high={res['high'][0]:.1f} (A->G mAP)")
print("PASS: high 明显 < low/mid (高频不可靠, confound 成立) → 建方法")
print("FAIL: 各带 mAP 接近 → 视角-频率不纠缠 → 判死换角度")
