#!/usr/bin/env python3
"""
IRRA fine-tuned encoder for the BSR kill-switch.

Wraps the IRRA (CVPR23, anosorae/IRRA) fine-tuned-on-RSTPReid checkpoint so it
exposes the SAME public API as FrozenCLIP in bsr_killswitch.py:

    encode_image(paths)  -> [N, D] L2-normalized
    encode_text(strs)    -> [N, D] L2-normalized

We reproduce IRRA's *retrieval* features exactly:
    image feat = base_model.encode_image(img)[:, 0, :]            # ViT CLS token
    text  feat = base_model.encode_text(tok)[arange, tok.argmax]  # EOT token

base_model is IRRA's CLIP ViT-B/16 with stride16 and 384x128 input. The CLIP
class config is auto-detected by build_CLIP_from_openai_pretrained from the
openai ViT-B/16 seed weights, then we OVERRIDE with the fine-tuned
`base_model.*` sub-state-dict (prefix stripped).

Tokenizer: IRRA's own SimpleTokenizer (it inserts <|mask|> and pops 'jekyll',
so its ids differ from openai clip.tokenize). EOT id = 49407 = max, which is
why text.argmax(-1) selects the EOT position.

Requires the IRRA repo on sys.path (--irra_repo, default /tmp/IRRA).
"""

import os
import sys
import types

import torch

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    print("PIL import failed:", e, file=sys.stderr)
    raise

import torchvision.transforms as T


# IRRA test-time preprocessing (datasets/build.py, is_train=False).
_IRRA_MEAN = [0.48145466, 0.4578275, 0.40821073]
_IRRA_STD = [0.26862954, 0.26130258, 0.27577711]


def _build_transform(img_size=(384, 128)):
    h, w = img_size
    return T.Compose([
        T.Resize((h, w)),
        T.ToTensor(),
        T.Normalize(mean=_IRRA_MEAN, std=_IRRA_STD),
    ])


class _Args:
    """Minimal args object for IRRA's CLIP submodule construction.

    We only need build_CLIP_from_openai_pretrained, so just the fields it reads
    plus img_size / stride_size.
    """
    pretrain_choice = "ViT-B/16"
    img_size = (384, 128)
    stride_size = 16


class IRRAEncoder:
    def __init__(self, ckpt_path, device="cuda", irra_repo="/tmp/IRRA",
                 img_size=(384, 128), stride_size=16):
        if irra_repo not in sys.path:
            sys.path.insert(0, irra_repo)

        # IRRA's clip_model.build_CLIP_from_openai_pretrained needs the openai
        # ViT-B/16 weights as a *config seed* (it reads shapes), then we load
        # our fine-tuned weights over it. The seed download is the standard
        # openai checkpoint to ~/.cache/clip.
        from model.clip_model import build_CLIP_from_openai_pretrained
        from utils.simple_tokenizer import SimpleTokenizer
        from datasets.bases import tokenize as irra_tokenize

        self.device = device
        self.img_size = img_size
        self._tokenize = irra_tokenize
        self.tokenizer = SimpleTokenizer()
        self.transform = _build_transform(img_size)

        # Build CLIP structure (config auto-detected from openai ViT-B/16) with
        # the ReID image size + stride. This already seeds with openai weights;
        # we override below with the fine-tuned ones.
        base_model, base_cfg = build_CLIP_from_openai_pretrained(
            _Args.pretrain_choice, img_size, stride_size)
        self.base_cfg = base_cfg

        # Load fine-tuned base_model.* weights (strip the 'base_model.' prefix).
        ck = torch.load(ckpt_path, map_location="cpu")
        sd = ck["model"] if "model" in ck else ck
        base_sd = {k[len("base_model."):]: v
                   for k, v in sd.items() if k.startswith("base_model.")}
        if not base_sd:
            raise RuntimeError(
                "No 'base_model.' keys in checkpoint %s (keys e.g. %s)"
                % (ckpt_path, list(sd.keys())[:5]))

        # load_param handles pos-embed resize defensively; here shapes already
        # match (vision pos emb [193,768] for 384x128/stride16, text [77,512]).
        missing_before = set(base_model.state_dict().keys())
        base_model.load_param(base_sd)
        loaded = [k for k in base_sd if k in base_model.state_dict()]
        skipped = [k for k in base_sd if k not in base_model.state_dict()]
        not_filled = missing_before - set(loaded)
        print("[IRRA] base_model keys: total=%d loaded=%d skipped(extra)=%d "
              "model-keys-not-in-ckpt=%d"
              % (len(base_model.state_dict()), len(loaded), len(skipped),
                 len(not_filled)))
        if skipped:
            print("[IRRA]   skipped ckpt keys (not in model):", skipped[:5],
                  "..." if len(skipped) > 5 else "")
        if not_filled:
            print("[IRRA]   model keys NOT overridden by ckpt (kept openai "
                  "seed!):", sorted(not_filled)[:10],
                  "..." if len(not_filled) > 10 else "")

        self.base_model = base_model.to(device).eval()
        # base_model is fp32 here (build does not convert_weights). Keep fp32 so
        # encode is deterministic and matches the BSR cos computation.
        self.base_model.float()

    @torch.no_grad()
    def encode_text(self, texts, batch_size=256):
        feats = []
        for k in range(0, len(texts), batch_size):
            chunk = texts[k:k + batch_size]
            toks = torch.stack([
                self._tokenize(t, self.tokenizer, text_length=77, truncate=True)
                for t in chunk
            ], 0).to(self.device)
            # CLIP.encode_text returns [B, L, D]; IRRA picks the EOT position.
            x = self.base_model.encode_text(toks)  # [B, L, D]
            f = x[torch.arange(x.shape[0]), toks.argmax(dim=-1)].float()
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
        return torch.cat(feats, 0)

    @torch.no_grad()
    def encode_image(self, img_paths, batch_size=128):
        feats = []
        for k in range(0, len(img_paths), batch_size):
            chunk = img_paths[k:k + batch_size]
            ims = torch.stack([
                self.transform(Image.open(p).convert("RGB")) for p in chunk
            ], 0).to(self.device)
            # CLIP.encode_image returns [B, 1+HW, D]; IRRA picks the CLS token.
            x = self.base_model.encode_image(ims)  # [B, L, D]
            f = x[:, 0, :].float()
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu())
        return torch.cat(feats, 0)
