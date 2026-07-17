#!/usr/bin/env python3
"""
BSR (Binding Swap Rate) kill-switch for the Binding-Ambiguity T2I-ReID angle.

Goal
----
Test whether the (literature-confirmed) CLIP bag-of-words / weak-binding confound
actually shows up in *person-ReID retrieval* on RSTPReid captions.

We take captions that contain >=2 distinct (color, garment) bindings, build a
"same-bag-of-words, swapped-binding" hard negative by swapping the two color
tokens (grey jacket + blue shirt -> blue jacket + grey shirt). The word *set*
is unchanged; only the binding (which color goes with which garment) flips.

For each (image, original_caption, swapped_caption):
    sim_orig = cos(img, encode(original))
    sim_swap = cos(img, encode(swapped))
BSR (original-preference rate) = mean(sim_orig > sim_swap)
margin = mean(sim_orig - sim_swap)

A random-negative control (sim vs a random *other* caption) gives the "easy"
baseline. If binding-swap preference rate << random-negative preference rate,
the model is encoding tokens-present more than binding-correct => the confound
holds for ReID retrieval.

Encoders
--------
- frozen CLIP ViT-B/16 (default; proves the generic confound)
- IRRA fine-tuned checkpoint (interface reserved; --encoder irra)

Single GPU. Uses the lab-3090-installed `clip` package.
"""

import argparse
import json
import os
import random
import re
import sys

import torch

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    print("PIL import failed:", e, file=sys.stderr)
    raise


# ----------------------------------------------------------------------------
# Vocabulary for binding extraction
# ----------------------------------------------------------------------------

# Garments / parts that can carry a color binding. Multi-word entries must be
# listed before their substrings so the regex matches the longest first.
GARMENTS = [
    "down jacket", "t-shirt", "tee shirt", "polo shirt", "dress shirt",
    "overcoat", "trench coat", "leather jacket", "puffer jacket",
    "jacket", "coat", "shirt", "sweater", "hoodie", "cardigan", "vest",
    "trousers", "pants", "jeans", "shorts", "slacks", "overalls",
    "skirt", "dress",
    "shoes", "sneakers", "boots", "sandals", "trainers",
    "bag", "backpack", "handbag", "satchel",
    "hat", "cap", "beanie", "scarf",
]

# Colors. Multi-word modifiers (dark/light) handled separately as optional
# prefix so "dark blue trousers" -> color="dark blue".
BASE_COLORS = [
    "black", "white", "red", "blue", "green", "yellow", "grey", "gray",
    "brown", "orange", "purple", "pink", "beige", "khaki", "navy",
    "maroon", "tan", "silver", "gold", "cream",
]
COLOR_PREFIX = ["dark", "light", "bright", "pale", "deep"]

# Normalize grey/gray to compare colors as "distinct".
def _canon_color(c):
    c = c.strip().lower()
    c = c.replace("gray", "grey")
    return c


# Build regex: optional prefix + base color, whitespace, garment.
_garment_alt = "|".join(re.escape(g) for g in sorted(GARMENTS, key=len, reverse=True))
_color_alt = "|".join(re.escape(c) for c in BASE_COLORS)
_prefix_alt = "|".join(re.escape(p) for p in COLOR_PREFIX)

# Group 1 = full color phrase (possibly "dark blue"), group 2 = garment.
# We allow an optional single prefix word. We require word boundaries.
BIND_RE = re.compile(
    r"\b((?:(?:%s)\s+)?(?:%s))\s+(%s)\b" % (_prefix_alt, _color_alt, _garment_alt),
    flags=re.IGNORECASE,
)


def extract_bindings(caption):
    """Return list of dicts {span, color_phrase, color_canon, garment} for each
    (color, garment) binding found, in order of appearance.

    The 'color_phrase' is the exact substring (case preserved) so we can do a
    safe in-place swap. We only treat the *color* portion as swappable, so we
    record the color sub-span within the match.
    """
    out = []
    for m in BIND_RE.finditer(caption):
        full = m.group(0)
        color_phrase = m.group(1)
        garment = m.group(2)
        # color sub-span = the matched color group within the full match
        c_start = m.start(1)
        c_end = m.end(1)
        out.append({
            "match_start": m.start(),
            "match_end": m.end(),
            "color_start": c_start,
            "color_end": c_end,
            "color_phrase": color_phrase,
            "color_canon": _canon_color(color_phrase),
            "garment": garment.lower(),
        })
    return out


def make_swapped_caption(caption, bindings):
    """Pick two bindings with DISTINCT canonical colors AND distinct garments,
    swap their color phrases in-place. Returns (swapped_caption, info) or
    (None, None) if no valid swap pair exists.

    Swap is done by string slicing on the two color sub-spans so the rest of
    the text (and the word set) is byte-for-byte preserved aside from the two
    color tokens trading places.
    """
    # Find a pair (i, j) with different canonical color and different garment.
    n = len(bindings)
    pair = None
    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = bindings[i], bindings[j]
            if bi["color_canon"] == bj["color_canon"]:
                continue
            if bi["garment"] == bj["garment"]:
                # same garment word twice -> swapping colors is ambiguous, skip
                continue
            pair = (i, j)
            break
        if pair:
            break
    if pair is None:
        return None, None

    i, j = pair
    bi, bj = bindings[i], bindings[j]
    # order spans so we can rebuild left-to-right
    a, b = (bi, bj) if bi["color_start"] < bj["color_start"] else (bj, bi)

    swapped = (
        caption[: a["color_start"]]
        + b["color_phrase"]
        + caption[a["color_end"] : b["color_start"]]
        + a["color_phrase"]
        + caption[b["color_end"] :]
    )
    info = {
        "garment_a": a["garment"], "color_a": a["color_phrase"],
        "garment_b": b["garment"], "color_b": b["color_phrase"],
    }
    return swapped, info


# ----------------------------------------------------------------------------
# Encoders
# ----------------------------------------------------------------------------

class FrozenCLIP:
    """Wraps openai CLIP. Provides encode_image(paths) and encode_text(strs),
    both L2-normalized."""

    def __init__(self, name="ViT-B/16", device="cuda"):
        import clip
        self.clip = clip
        self.device = device
        self.model, self.preprocess = clip.load(name, device=device)
        self.model.eval()
        self.context_length = self.model.context_length

    @torch.no_grad()
    def encode_text(self, texts, batch_size=256):
        feats = []
        for k in range(0, len(texts), batch_size):
            chunk = texts[k:k + batch_size]
            tok = self.clip.tokenize(chunk, truncate=True).to(self.device)
            f = self.model.encode_text(tok).float()
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
        return torch.cat(feats, 0)

    @torch.no_grad()
    def encode_image(self, img_paths, batch_size=128):
        feats = []
        for k in range(0, len(img_paths), batch_size):
            chunk = img_paths[k:k + batch_size]
            ims = []
            for p in chunk:
                ims.append(self.preprocess(Image.open(p).convert("RGB")))
            ims = torch.stack(ims, 0).to(self.device)
            f = self.model.encode_image(ims).float()
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
        return torch.cat(feats, 0)


# IRRAEncoder lives in irra_encoder.py (needs the IRRA repo on sys.path). It
# mirrors FrozenCLIP's public API (encode_image / encode_text -> L2-normalized),
# so the BSR logic below stays encoder-agnostic. Imported lazily in main() only
# when --encoder irra is selected (keeps the frozen-CLIP path dependency-free).


# ----------------------------------------------------------------------------
# Main BSR evaluation
# ----------------------------------------------------------------------------

def build_swap_samples(records, img_root, split, max_samples=None):
    """For each record in the chosen split, for each caption, try to build a
    swapped negative. Returns parallel lists.
    """
    samples = []
    for rec in records:
        if split != "all" and rec.get("split") != split:
            continue
        img_path = os.path.join(img_root, rec["img_path"])
        if not os.path.exists(img_path):
            continue
        for cap in rec["captions"]:
            cap = cap.strip()
            binds = extract_bindings(cap)
            if len(binds) < 2:
                continue
            swapped, info = make_swapped_caption(cap, binds)
            if swapped is None or swapped == cap:
                continue
            samples.append({
                "img_path": img_path,
                "orig": cap,
                "swap": swapped,
                "info": info,
            })
    if max_samples is not None and len(samples) > max_samples:
        random.shuffle(samples)
        samples = samples[:max_samples]
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default="data/RSTPReid/data_captions.json")
    ap.add_argument("--img_root", default="data/RSTPReid/imgs")
    ap.add_argument("--split", default="test",
                    choices=["train", "val", "test", "all"])
    ap.add_argument("--encoder", default="clip", choices=["clip", "irra"])
    ap.add_argument("--clip_name", default="ViT-B/16")
    ap.add_argument("--irra_ckpt", default="irra_rstp/best.pth",
                    help="path to IRRA checkpoint .pth (or its dir; best.pth "
                         "is appended for a dir)")
    ap.add_argument("--irra_repo", default="/tmp/IRRA",
                    help="path to cloned anosorae/IRRA repo (for model code + "
                         "tokenizer)")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="cap number of swap samples (None = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_examples", type=int, default=3)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.captions) as f:
        records = json.load(f)

    samples = build_swap_samples(records, args.img_root, args.split,
                                 args.max_samples)
    if not samples:
        print("No swappable samples found. Check vocab / data.", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("BSR kill-switch | encoder=%s | split=%s | device=%s"
          % (args.encoder, args.split, device))
    print("swappable samples: %d" % len(samples))
    print("=" * 70)

    # Build encoder
    if args.encoder == "clip":
        enc = FrozenCLIP(args.clip_name, device=device)
    else:
        from irra_encoder import IRRAEncoder
        ckpt = args.irra_ckpt
        if os.path.isdir(ckpt):
            ckpt = os.path.join(ckpt, "best.pth")
        enc = IRRAEncoder(ckpt, device=device, irra_repo=args.irra_repo)

    img_paths = [s["img_path"] for s in samples]
    orig_caps = [s["orig"] for s in samples]
    swap_caps = [s["swap"] for s in samples]

    # Random negative: a shuffled derangement over the *original* captions so
    # each image is paired with some OTHER image's original caption.
    n = len(samples)
    perm = list(range(n))
    random.shuffle(perm)
    for i in range(n):  # fix accidental fixed points -> true derangement
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    rand_caps = [orig_caps[perm[i]] for i in range(n)]

    print("[encode] images ...", flush=True)
    img_f = enc.encode_image(img_paths)
    print("[encode] original captions ...", flush=True)
    orig_f = enc.encode_text(orig_caps)
    print("[encode] swapped captions ...", flush=True)
    swap_f = enc.encode_text(swap_caps)
    print("[encode] random-neg captions ...", flush=True)
    rand_f = enc.encode_text(rand_caps)

    # Cosine sims (feats already L2-normalized) -> row-wise dot product.
    sim_orig = (img_f * orig_f).sum(-1)
    sim_swap = (img_f * swap_f).sum(-1)
    sim_rand = (img_f * rand_f).sum(-1)

    bsr = (sim_orig > sim_swap).float().mean().item()
    margin_swap = (sim_orig - sim_swap).mean().item()
    rand_win = (sim_orig > sim_rand).float().mean().item()
    margin_rand = (sim_orig - sim_rand).mean().item()

    # ties (exactly equal) diagnostic
    ties_swap = (sim_orig == sim_swap).float().mean().item()

    print()
    print("-" * 70)
    print("RESULTS  (encoder=%s, n=%d)" % (args.encoder, n))
    print("-" * 70)
    print("BSR (orig-preference vs SWAPPED) : %.4f" % bsr)
    print("  mean margin (orig - swap)      : %+.5f" % margin_swap)
    print("  exact-tie rate                 : %.4f" % ties_swap)
    print("Random-neg orig-preference rate  : %.4f" % rand_win)
    print("  mean margin (orig - random)    : %+.5f" % margin_rand)
    print("-" * 70)
    gap = rand_win - bsr
    print("GAP (random_win - BSR)           : %+.4f" % gap)
    print()

    # Verdict heuristic for frozen CLIP
    print("INTERPRETATION:")
    if rand_win < 0.80:
        print("  ! random-neg win-rate is low (%.2f); retrieval signal weak,"
              " BSR comparison less conclusive." % rand_win)
    if bsr < rand_win - 0.05:
        print("  -> BSR (%.2f) clearly BELOW random-neg win (%.2f):"
              " swapped-binding negatives are much harder than random text"
              " => binding confound HOLDS (not generic text noise)." % (bsr, rand_win))
    elif bsr < rand_win:
        print("  -> BSR (%.2f) modestly below random-neg win (%.2f):"
              " weak binding sensitivity." % (bsr, rand_win))
    else:
        print("  -> BSR (%.2f) >= random-neg win (%.2f): model already"
              " prefers correct binding => confound NOT evident here." % (bsr, rand_win))

    # Examples
    print()
    print("EXAMPLES (orig vs swapped, sim_orig / sim_swap):")
    show = list(range(len(samples)))
    random.shuffle(show)
    shown = 0
    for idx in show:
        if shown >= args.n_examples:
            break
        s = samples[idx]
        print("  [%d] sim_orig=%.4f  sim_swap=%.4f  margin=%+.4f"
              % (idx, sim_orig[idx], sim_swap[idx], sim_orig[idx] - sim_swap[idx]))
        print("      orig: %s" % s["orig"])
        print("      swap: %s" % s["swap"])
        print("      pair: %s<->%s on (%s / %s)"
              % (s["info"]["color_a"], s["info"]["color_b"],
                 s["info"]["garment_a"], s["info"]["garment_b"]))
        shown += 1

    print()
    print("DONE.")


if __name__ == "__main__":
    main()
