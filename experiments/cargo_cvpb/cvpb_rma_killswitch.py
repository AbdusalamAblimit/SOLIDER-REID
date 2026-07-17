#!/usr/bin/env python3
"""RMA-TIReID  —  ZERO-TRAINING kill-switch (frozen SOLIDER + numpy only).

Re-frame under test (pivot/clean/ondisk_pivot.txt):
    text-to-image ReID fails NOT because "local word-patch alignment is off", but
    because an UNDER-COMPLETE text query does not LAND on the correct region of a
    STRONG visual ReID identity manifold. Team asset = a strong person-ReID image
    encoder (SOLIDER/Swin, exp260b market mAP 94.4). RSTPReid person crops are the
    SAME domain (person crops), so the frozen image encoder should give a good
    image-image identity manifold even though it never saw RSTPReid text.

Tests (all frozen, no_grad, numpy):
  A. image-image identity manifold quality on RSTPReid test (i2i mAP/R1). If the
     manifold is good, same-ID images are highly retrievable in frozen SOLIDER space.
  B. zero-training token->visual-prototype anchored TEXT query: build a
     token -> visual-direction table from TRAIN captions+images (mean feature of
     train images whose caption contains the token, AND a present-minus-absent
     residual variant). A test text query = IDF-weighted sum of its tokens'
     prototypes -> query the frozen gallery image features (t2i mAP/R1).
  Controls (decide life/death):
     - color-only query (only color tokens)         -> "does it just read color?"
     - token-shuffle  (permute token->prototype map) -> destroys token semantics
     - feature-shuffle(permute gallery feat<->id)    -> destroys manifold
     - random-prototype (random unit vectors per tok)-> pure chance ceiling
     token-prototype MUST clearly beat color-only AND all shuffles, else DEAD.
  C. re-frame validation: for TEXT queries that the token-prototype FAILS (rank-1
     wrong / low AP), is the TARGET image still image-image retrievable? If yes in
     a high fraction, the failure is "text did not land on the manifold", not "the
     manifold is missing the identity" — that is the re-frame's core claim.

Run on lab-3090-d:
    cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
      /root/miniconda3/envs/solider-reid/bin/python \
      experiments/cargo_cvpb/cvpb_rma_killswitch.py \
      --config configs/market/pose_psg_lgpa_gcn_base.yml \
      --ckpt   log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth \
      --rstp   data/RSTPReid \
      --cache_feat /tmp/rma_rstp_feats.npz 2>&1 | tee experiments/cargo_cvpb/cvpb_rma.log
"""
import os, sys, re, json, time, argparse
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.abspath(os.path.join(_here, '..', '..'))   # repo root .../SOLIDER-REID
sys.path.insert(0, _repo)

ap = argparse.ArgumentParser()
ap.add_argument('--config', default='configs/market/pose_psg_lgpa_gcn_base.yml')
ap.add_argument('--ckpt',   default='log/market1501/exp260b_base_gcn512_2stage/transformer_120.pth')
ap.add_argument('--rstp',   default='data/RSTPReid')
ap.add_argument('--cache_feat', default='/tmp/rma_rstp_feats.npz')
ap.add_argument('--reuse_feat', action='store_true')
ap.add_argument('--min_token_freq', type=int, default=30,
                help='min #train-images a token must appear in to get a prototype')
ap.add_argument('--residual', action='store_true',
                help='token prototype = mean(present) - mean(absent) instead of mean(present)')
ap.add_argument('--seed', type=int, default=42)
cli = ap.parse_args()
np.random.seed(cli.seed)
RNG = np.random.RandomState(cli.seed)


# =========================================================================== #
# RSTPReid attribute-token vocabulary (reused/extended from cbcl_t2i bsr).
# =========================================================================== #
GARMENTS = [
    "down jacket", "t-shirt", "tee shirt", "polo shirt", "dress shirt",
    "overcoat", "trench coat", "leather jacket", "puffer jacket",
    "jacket", "coat", "shirt", "sweater", "hoodie", "cardigan", "vest",
    "trousers", "pants", "jeans", "shorts", "slacks", "overalls",
    "skirt", "dress",
    "shoes", "sneakers", "boots", "sandals", "trainers",
    "bag", "backpack", "handbag", "satchel",
    "hat", "cap", "beanie", "scarf", "glasses", "mask", "umbrella",
]
BASE_COLORS = [
    "black", "white", "red", "blue", "green", "yellow", "grey", "gray",
    "brown", "orange", "purple", "pink", "beige", "khaki", "navy",
    "maroon", "tan", "silver", "gold", "cream",
]
COLOR_PREFIX = ["dark", "light", "bright", "pale", "deep"]
# Non-color attribute tokens worth their own prototype (gender / accessory / length).
EXTRA_TOKENS = [
    "man", "woman", "male", "female", "boy", "girl", "lady", "guy",
    "backpack", "handbag", "shoulder bag", "long hair", "short hair",
    "long sleeve", "short sleeve", "long sleeves", "short sleeves",
    "glasses", "hat", "cap", "mask", "umbrella",
]
COLOR_SET = set(c.replace("gray", "grey") for c in BASE_COLORS)


def canon_color(c):
    return c.strip().lower().replace("gray", "grey")


def parse_tokens(caption):
    """Return a set of attribute tokens for a caption.
       - color tokens:           e.g. "black", "grey" (canonicalized)
       - color+garment bindings: e.g. "black trousers", "blue shirt"
       - bare garments:          e.g. "backpack", "dress"
       - extra tokens:           e.g. "woman", "long sleeve"
    Multi-word garments / extras matched longest-first."""
    s = caption.lower()
    toks = set()
    # extras (longest first so "shoulder bag" before "bag" not needed here)
    for e in sorted(EXTRA_TOKENS, key=len, reverse=True):
        if re.search(r'\b' + re.escape(e) + r'\b', s):
            toks.add(e)
    # color (+ optional prefix) garment bindings, longest garment first
    color_alt = '|'.join(sorted([canon_color(c) for c in BASE_COLORS] + ['gray'], key=len, reverse=True))
    prefix_alt = '|'.join(COLOR_PREFIX)
    for g in sorted(GARMENTS, key=len, reverse=True):
        # "<optional prefix> <color> ... <garment>" with up to 2 filler words between color and garment
        pat = r'\b(?:(?:' + prefix_alt + r')\s+)?(' + color_alt + r')\b(?:\s+\w+){0,2}?\s+' + re.escape(g) + r'\b'
        for m in re.finditer(pat, s):
            col = canon_color(m.group(1))
            toks.add(col)                       # the color itself
            toks.add(g)                         # the bare garment
            toks.add(f'{col} {g}')              # the binding
    # bare colors anywhere (so "in black" still contributes)
    for c in BASE_COLORS:
        cc = canon_color(c)
        if re.search(r'\b' + cc + r'\b', s):
            toks.add(cc)
    # bare garments anywhere
    for g in GARMENTS:
        if re.search(r'\b' + re.escape(g) + r'\b', s):
            toks.add(g)
    return toks


def is_color_token(t):
    """color token OR color+garment binding whose head is a color."""
    if t in COLOR_SET:
        return True
    head = t.split(' ')[0]
    return head in COLOR_SET


# =========================================================================== #
# 1. FEATURE EXTRACTION  (frozen ckpt, global BN-neck vector, no pose)
# =========================================================================== #
def _build_model():
    import torch
    from config import cfg
    from model import make_model
    cfg.merge_from_file(os.path.join(_repo, cli.config))
    cfg.merge_from_list([
        'TEST.WEIGHT', os.path.join(_repo, cli.ckpt),
        'MODEL.POSE_TEST_FEAT', 'global',   # single global vector, no pose part-branch
        'TEST.NECK_FEAT', 'after',          # BN-neck (trained eval feature)
        'TEST.FEAT_NORM', 'yes',
        'TEST.IMS_PER_BATCH', 64,
    ])
    cfg.freeze()
    # exp260b trained on market1501 (751 train ids, 6 cams). classifier head unused at eval.
    model = make_model(cfg, num_class=751, camera_num=6, view_num=1,
                       semantic_weight=cfg.MODEL.SEMANTIC_WEIGHT)
    model.load_param(cfg.TEST.WEIGHT)
    model = model.cuda().eval()
    return model, cfg


def _extract_paths(model, paths, bs=64):
    """Extract L2-normalized global features for a list of absolute image paths."""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import torchvision.transforms as T
    tf = T.Compose([T.Resize((384, 128)),
                    T.ToTensor(),
                    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    feats = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            chunk = paths[i:i + bs]
            imgs = torch.stack([tf(Image.open(p).convert('RGB')) for p in chunk]).cuda(non_blocking=True)
            n = imgs.shape[0]
            out = model(imgs,
                        cam_label=torch.zeros(n, dtype=torch.long).cuda(),
                        view_label=torch.zeros(n, dtype=torch.long).cuda())
            feat = out[0] if isinstance(out, (tuple, list)) else out
            assert torch.is_tensor(feat) and feat.dim() == 2, \
                f"expected (B,D) global feat, got {getattr(feat,'shape',None)}"
            feat = F.normalize(feat, p=2, dim=1)
            feats.append(feat.cpu().numpy().astype(np.float32))
            if (i // bs) % 20 == 0:
                print(f"  [extract] {i}/{len(paths)} ({time.time()-t0:.0f}s)", flush=True)
    return np.concatenate(feats, 0)


def load_rstp():
    """Parse data_captions.json. Returns dict with per-split image records and the
    raw caption list. person id = img_path prefix (e.g. '3901_c14_0007.jpg' -> 3901)."""
    jpath = os.path.join(_repo, cli.rstp, 'data_captions.json')
    data = json.load(open(jpath))
    img_root = os.path.join(_repo, cli.rstp, 'imgs')
    def pid_of(p): return int(p.split('_')[0])
    def cam_of(p): return int(re.search(r'_c(\d+)_', p).group(1))
    rec = {'train': [], 'val': [], 'test': []}
    for e in data:
        sp = e['split']
        rec[sp].append(dict(
            path=os.path.join(img_root, e['img_path']),
            pid=pid_of(e['img_path']),
            cam=cam_of(e['img_path']),
            captions=e['captions'],
        ))
    return rec


def get_features(rec):
    if cli.reuse_feat and os.path.exists(cli.cache_feat):
        z = np.load(cli.cache_feat, allow_pickle=True)
        print(f"[reuse] features from {cli.cache_feat}")
        return (z['test_feat'], z['test_pid'], z['test_cam'],
                z['train_feat'], z['train_pid'])
    model, _ = _build_model()
    print("[extract] TEST split features ...", flush=True)
    test_paths = [r['path'] for r in rec['test']]
    test_feat = _extract_paths(model, test_paths)
    test_pid = np.array([r['pid'] for r in rec['test']])
    test_cam = np.array([r['cam'] for r in rec['test']])
    print("[extract] TRAIN split features (for token prototypes) ...", flush=True)
    train_paths = [r['path'] for r in rec['train']]
    train_feat = _extract_paths(model, train_paths)
    train_pid = np.array([r['pid'] for r in rec['train']])
    np.savez(cli.cache_feat,
             test_feat=test_feat, test_pid=test_pid, test_cam=test_cam,
             train_feat=train_feat, train_pid=train_pid)
    print(f"[extract] cached -> {cli.cache_feat}", flush=True)
    return test_feat, test_pid, test_cam, train_feat, train_pid


# =========================================================================== #
# 2. EVAL (id-level; t2i: query=text, gallery=image / i2i: query=image, gallery=image)
# =========================================================================== #
def eval_rank(sim, q_pid, g_pid, q_cam=None, g_cam=None, drop_same=False, max_rank=20):
    """sim: (Nq,Ng) cosine (higher=closer). A gallery hit is correct if same pid.
    If drop_same (i2i leave-one-out): drop gallery with same pid&cam as the query
    (market-style junk removal). Returns mAP / R1 / R5 / R10 and per-query AP."""
    num_q = sim.shape[0]
    order = np.argsort(-sim, axis=1)
    aps = np.full(num_q, -1.0)
    cmcs = []
    for i in range(num_q):
        o = order[i]
        if drop_same:
            keep = ~((g_pid[o] == q_pid[i]) & (g_cam[o] == q_cam[i]))
            o = o[keep]
        m = (g_pid[o] == q_pid[i]).astype(np.int32)
        if not m.any():
            continue
        cmc = m.cumsum(); cmc[cmc > 1] = 1
        cmcs.append(cmc[:max_rank])
        tmp = m.cumsum()
        prec = tmp / (np.arange(len(m)) + 1.0)
        aps[i] = (prec * m).sum() / m.sum()
    cmcs = np.asarray(cmcs).mean(0)
    valid = aps >= 0
    return dict(mAP=float(np.mean(aps[valid])) * 100,
                r1=float(cmcs[0]) * 100, r5=float(cmcs[4]) * 100, r10=float(cmcs[9]) * 100,
                nq=int(valid.sum())), aps


# =========================================================================== #
# 3. TOKEN -> VISUAL-PROTOTYPE TABLE (from TRAIN captions+images)
# =========================================================================== #
def build_token_table(rec_train, train_feat):
    """For each token appearing in >= min_token_freq TRAIN images, build a visual
    prototype = mean (L2-normed) feature of train images whose ANY caption contains
    the token. Residual variant = mean(present) - mean(absent). Also return per-token
    document frequency (for IDF) over train images."""
    Ntr = len(rec_train)
    # token -> set of train-image indices containing it (in any of its captions)
    tok2idx = {}
    img_tokens = []
    for i, r in enumerate(rec_train):
        toks = set()
        for cap in r['captions']:
            toks |= parse_tokens(cap)
        img_tokens.append(toks)
        for t in toks:
            tok2idx.setdefault(t, set()).add(i)
    # keep frequent tokens
    kept = {t: idx for t, idx in tok2idx.items() if len(idx) >= cli.min_token_freq}
    proto = {}
    df = {}
    tf = train_feat
    tf = tf / (np.linalg.norm(tf, axis=1, keepdims=True) + 1e-12)
    allmean = tf.mean(0)
    for t, idxset in kept.items():
        idx = np.array(sorted(idxset))
        present = tf[idx].mean(0)
        if cli.residual:
            mask = np.ones(Ntr, bool); mask[idx] = False
            absent = tf[mask].mean(0) if mask.any() else allmean
            v = present - absent
        else:
            v = present
        n = np.linalg.norm(v) + 1e-12
        proto[t] = (v / n).astype(np.float32)
        df[t] = len(idxset)
    idf = {t: np.log(Ntr / df[t]) for t in kept}
    print(f"[tokens] {len(tok2idx)} raw tokens, {len(kept)} kept (freq>={cli.min_token_freq}). "
          f"residual={cli.residual}")
    # show a few high/low frequency kept tokens
    top = sorted(df.items(), key=lambda kv: -kv[1])[:18]
    print("[tokens] top kept (token: train-img-freq):")
    for t, c in top:
        print(f"    {t:22s} {c:5d}  IDF={idf[t]:.2f}  {'[COLOR]' if is_color_token(t) else ''}")
    return proto, idf


def text_query_vec(caption, proto, idf, color_only=False, token_subset=None):
    """IDF-weighted sum of token prototypes for one caption. Returns None if no
    usable token. color_only -> keep only color tokens. token_subset (dict) ->
    use this prototype map instead of `proto` (for shuffle controls)."""
    P = token_subset if token_subset is not None else proto
    toks = parse_tokens(caption)
    vecs, ws = [], []
    for t in toks:
        if t not in P:
            continue
        if color_only and not is_color_token(t):
            continue
        vecs.append(P[t]); ws.append(idf.get(t, 1.0))
    if not vecs:
        return None
    V = np.array(vecs); w = np.array(ws)[:, None]
    q = (V * w).sum(0)
    n = np.linalg.norm(q) + 1e-12
    return (q / n).astype(np.float32)


def build_text_queries(rec_test, proto, idf, color_only=False, token_subset=None):
    """One query per caption (2 per test image). Returns (Q, q_pid, valid_mask, owner_img)
    where owner_img is the index into rec_test of the source image."""
    Q, qp, owner = [], [], []
    for i, r in enumerate(rec_test):
        for cap in r['captions']:
            v = text_query_vec(cap, proto, idf, color_only=color_only, token_subset=token_subset)
            Q.append(v if v is not None else np.zeros(0, np.float32))
            qp.append(r['pid']); owner.append(i)
    return Q, np.array(qp), np.array(owner)


def _pack(Q):
    """Pack a list of (D,)/empty vectors into (N,D) with a valid mask for non-empty."""
    D = next((v.shape[0] for v in Q if v.size), 0)
    M = np.zeros((len(Q), D), np.float32)
    valid = np.zeros(len(Q), bool)
    for i, v in enumerate(Q):
        if v.size:
            M[i] = v; valid[i] = True
    return M, valid


def run_t2i(rec_test, gf, g_pid, proto, idf, label, color_only=False, token_subset=None):
    Q, qp, owner = build_text_queries(rec_test, proto, idf,
                                      color_only=color_only, token_subset=token_subset)
    M, valid = _pack(Q)
    if valid.sum() == 0:
        print(f"  [{label}] NO valid queries"); return None, None, None, None
    sim = M[valid] @ gf.T
    res, aps_v = eval_rank(sim, qp[valid], g_pid)
    # map per-query AP back to full caption-index space
    aps = np.full(len(Q), -1.0); aps[np.where(valid)[0]] = aps_v
    print(f"  [{label:22s}] t2i  mAP={res['mAP']:5.2f}  R1={res['r1']:5.2f}  "
          f"R5={res['r5']:5.2f}  R10={res['r10']:5.2f}  (nq_valid={res['nq']}/{len(Q)})")
    return res, aps, qp, owner


# =========================================================================== #
# MAIN
# =========================================================================== #
def main():
    print("#" * 80)
    print("# RMA-TIReID ZERO-TRAINING KILL-SWITCH  (frozen SOLIDER exp260b, RSTPReid)")
    print("#" * 80)
    rec = load_rstp()
    print(f"[data] train={len(rec['train'])} val={len(rec['val'])} test={len(rec['test'])} images; "
          f"test ids={len(set(r['pid'] for r in rec['test']))}")

    test_feat, test_pid, test_cam, train_feat, train_pid = get_features(rec)
    gf = test_feat / (np.linalg.norm(test_feat, axis=1, keepdims=True) + 1e-12)
    g_pid, g_cam = test_pid, test_cam
    print(f"[data] test gallery={gf.shape[0]} imgs dim={gf.shape[1]}; "
          f"train pool={train_feat.shape[0]} imgs")

    # ====================================================================== #
    # TEST A — image-image identity manifold quality (the re-frame's premise)
    # ====================================================================== #
    print("\n" + "=" * 80)
    print("TEST A — image-image identity manifold on RSTPReid test (frozen SOLIDER)")
    print("=" * 80)
    sim_ii = gf @ gf.T
    np.fill_diagonal(sim_ii, -2.0)   # never retrieve self
    res_ii, aps_ii = eval_rank(sim_ii, g_pid, g_pid, q_cam=g_cam, g_cam=g_cam,
                               drop_same=True)
    print(f"  i2i (leave-one-out, drop same id+cam):  "
          f"mAP={res_ii['mAP']:.2f}  R1={res_ii['r1']:.2f}  R5={res_ii['r5']:.2f}  "
          f"R10={res_ii['r10']:.2f}  (nq={res_ii['nq']})")
    print(f"  >> manifold quality: high i2i mAP/R1 == same-ID images form a tight, "
          f"retrievable neighborhood (re-frame premise holds if this is strong).")

    # ====================================================================== #
    # TEST B — token->visual-prototype anchored TEXT query  + controls
    # ====================================================================== #
    print("\n" + "=" * 80)
    print("TEST B — zero-training token-prototype text->image retrieval + controls")
    print("=" * 80)
    proto, idf = build_token_table(rec['train'], train_feat)

    print("\n  --- main token-prototype query (IDF-weighted, all tokens) ---")
    res_full, aps_full, qp_full, owner_full = run_t2i(
        rec['test'], gf, g_pid, proto, idf, 'token-proto (ALL)')

    print("\n  --- CONTROLS (life/death) ---")
    # color-only
    res_color, aps_color, _, _ = run_t2i(
        rec['test'], gf, g_pid, proto, idf, 'color-only', color_only=True)
    # token-shuffle: permute the token->prototype mapping (keep same set of prototypes,
    # assign each to a WRONG token). Destroys token semantics; keeps prototype geometry.
    toks = list(proto.keys())
    perm = RNG.permutation(len(toks))
    proto_shuf = {toks[i]: proto[toks[perm[i]]] for i in range(len(toks))}
    res_tsh, aps_tsh, _, _ = run_t2i(
        rec['test'], gf, g_pid, proto_shuf, idf, 'token-shuffle', token_subset=proto_shuf)
    # random-prototype: replace each token prototype with a fixed random unit vector.
    D = gf.shape[1]
    proto_rand = {t: (lambda v: v / (np.linalg.norm(v) + 1e-12))(RNG.randn(D).astype(np.float32))
                  for t in proto}
    res_rand, aps_rand, _, _ = run_t2i(
        rec['test'], gf, g_pid, proto_rand, idf, 'random-prototype', token_subset=proto_rand)
    # feature-shuffle: permute gallery feature<->id binding (destroys the manifold the
    # query is supposed to land on). Use the REAL token-proto queries vs shuffled gallery.
    print("  --- feature-shuffle control (shuffle gallery feat<->id) ---")
    Q, qp, owner = build_text_queries(rec['test'], proto, idf)
    M, valid = _pack(Q)
    gperm = RNG.permutation(gf.shape[0])
    sim_fs = M[valid] @ gf[gperm].T          # gallery features permuted vs their pids
    res_fs, _ = eval_rank(sim_fs, qp[valid], g_pid)   # g_pid NOT permuted -> binding broken
    print(f"  [feature-shuffle        ] t2i  mAP={res_fs['mAP']:5.2f}  R1={res_fs['r1']:5.2f}  "
          f"(expect ~chance)")

    # chance reference (uniform): mAP for random ranking ~ mean over q of (#pos/Ng)*correction;
    # report the empirical "random query" too: random unit query vectors.
    Rq = RNG.randn(valid.sum(), D).astype(np.float32)
    Rq /= (np.linalg.norm(Rq, axis=1, keepdims=True) + 1e-12)
    res_rq, _ = eval_rank(Rq @ gf.T, qp[valid], g_pid)
    print(f"  [random-query           ] t2i  mAP={res_rq['mAP']:5.2f}  R1={res_rq['r1']:5.2f}  "
          f"(pure chance ceiling)")

    # ====================================================================== #
    # VERDICT lines for B
    # ====================================================================== #
    print("\n  --- B verdict deltas ---")
    g_full = res_full['mAP'] if res_full else float('nan')
    print(f"  token-proto(ALL) mAP={g_full:.2f}")
    print(f"    vs color-only      {res_color['mAP']:.2f}  (delta {g_full-res_color['mAP']:+.2f})")
    print(f"    vs token-shuffle   {res_tsh['mAP']:.2f}  (delta {g_full-res_tsh['mAP']:+.2f})")
    print(f"    vs random-proto    {res_rand['mAP']:.2f}  (delta {g_full-res_rand['mAP']:+.2f})")
    print(f"    vs feature-shuffle {res_fs['mAP']:.2f}  (delta {g_full-res_fs['mAP']:+.2f})")
    print(f"    vs random-query    {res_rq['mAP']:.2f}  (delta {g_full-res_rq['mAP']:+.2f})")
    beat_color = g_full - res_color['mAP']
    beat_shuf = g_full - max(res_tsh['mAP'], res_rand['mAP'], res_fs['mAP'], res_rq['mAP'])
    print(f"  >> CORE: token-proto must CLEARLY beat color-only (delta={beat_color:+.2f}) "
          f"AND all shuffles (delta_vs_best_shuffle={beat_shuf:+.2f}).")

    # ====================================================================== #
    # TEST C — re-frame validation: failed TEXT queries, is target image i2i-retrievable?
    # ====================================================================== #
    print("\n" + "=" * 80)
    print("TEST C — re-frame: token-proto-FAILED text queries -> is target image i2i-retrievable?")
    print("=" * 80)
    # per-image i2i retrievability: AP_ii (already have aps_ii indexed by test image),
    # and i2i rank-1 hit per image.
    order_ii = np.argsort(-sim_ii, axis=1)
    r1_ii = np.zeros(gf.shape[0], bool)
    for i in range(gf.shape[0]):
        o = order_ii[i]
        keep = ~((g_pid[o] == g_pid[i]) & (g_cam[o] == g_cam[i]))
        o = o[keep]
        if len(o):
            r1_ii[i] = (g_pid[o[0]] == g_pid[i])

    # token-proto t2i: per-caption rank-1 hit and AP, owner image index.
    Qf, qp_f, owner_f = build_text_queries(rec['test'], proto, idf)
    Mf, validf = _pack(Qf)
    sim_t = Mf[validf] @ gf.T
    order_t = np.argsort(-sim_t, axis=1)
    qp_v = qp_f[validf]; owner_v = owner_f[validf]
    t2i_r1 = (g_pid[order_t[:, 0]] == qp_v)              # caption rank-1 correct?
    # per-caption AP for the threshold on "low-AP" failures
    aps_t = aps_full[validf] if aps_full is not None else None

    # Define FAILED text query = rank-1 wrong (strict) OR AP below 10% (soft).
    failed_r1 = ~t2i_r1
    failed_ap = (aps_t < 0.10) if aps_t is not None else failed_r1
    for fname, failed in [('rank-1-wrong', failed_r1), ('AP<10%', failed_ap)]:
        own = owner_v[failed]
        if len(own) == 0:
            print(f"  [{fname}] no failed queries"); continue
        tgt_r1_ii = r1_ii[own]          # is the TARGET image's i2i rank-1 correct?
        tgt_ap_ii = aps_ii[own]
        frac_r1 = float(tgt_r1_ii.mean()) * 100
        frac_apgt = float((tgt_ap_ii >= 0.30).mean()) * 100   # target i2i AP>=30%
        print(f"  [{fname:13s}] #failed-text-q={len(own):4d}  "
              f"target-image i2i-R1-retrievable = {frac_r1:5.1f}%   "
              f"target-image i2i-AP>=30% = {frac_apgt:5.1f}%   "
              f"(mean target i2i AP={100*np.nanmean(tgt_ap_ii[tgt_ap_ii>=0]):.1f})")
    print(f"  >> re-frame holds if FAILED text queries still have HIGH target-image i2i "
          f"retrievability (image manifold has the identity; text did not land on it).")

    # ====================================================================== #
    # FINAL SUMMARY
    # ====================================================================== #
    print("\n" + "#" * 80)
    print("SUMMARY / VERDICT  (RSTPReid, frozen SOLIDER exp260b)")
    print("#" * 80)
    print(f"  A  i2i manifold mAP/R1                 = {res_ii['mAP']:.2f} / {res_ii['r1']:.2f}")
    print(f"  B  token-proto t2i mAP/R1             = {g_full:.2f} / {res_full['r1']:.2f}")
    print(f"     color-only / token-shuf / rand-proto = "
          f"{res_color['mAP']:.2f} / {res_tsh['mAP']:.2f} / {res_rand['mAP']:.2f}")
    print(f"     feature-shuffle / random-query     = {res_fs['mAP']:.2f} / {res_rq['mAP']:.2f}")
    print(f"     beat color-only / beat best-shuffle= {beat_color:+.2f} / {beat_shuf:+.2f}")
    print(f"  C  see TEST C above (failed-text target i2i retrievability)")
    print("\n  DECISION RULE:")
    print("   - token-proto >> color-only AND >> shuffles  -> anchoring is real (proceed-ish)")
    print("   - token-proto ~= color-only OR ~= shuffles    -> DEAD (only reads color / chance)")
    print("\n[done] RMA-TIReID kill-switch complete.")


if __name__ == '__main__':
    main()
