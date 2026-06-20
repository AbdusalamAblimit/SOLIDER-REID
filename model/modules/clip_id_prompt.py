"""CLIP-ReID-style learnable ID text prompts (CoOp) for SOLIDER.

The PROVEN CLIP-in-ReID mechanism: per-ID learnable context tokens encoded by the
frozen CLIP text transformer into ID text prototypes, aligned to image features via
i2t/t2i contrastive. Unlike fixed body-part text prototypes (exp340 = shell), these
prototypes are LEARNED per identity from data — the mechanism that actually gains.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

# "A photo of a [X X X X] person." — 4 fixed ctx ("A photo of a") + 4 learnable per-ID + "person."
_TEMPLATE = "A photo of a X X X X person."
_N_CTX = 4          # tokens in "A photo of a"
_N_CLS_CTX = 4      # learnable per-ID context tokens (the X X X X)


class CLIPIDPromptLearner(nn.Module):
    def __init__(self, num_classes, clip_arch='ViT-B-32', clip_pretrained='openai',
                 pose_cond=False, pose_dim=17):
        super().__init__()
        clip_model, _, _ = open_clip.create_model_and_transforms(clip_arch, pretrained=clip_pretrained)
        tokenizer = open_clip.get_tokenizer(clip_arch)
        ctx_dim = clip_model.token_embedding.weight.shape[1]
        self.ctx_dim = ctx_dim
        dtype = clip_model.token_embedding.weight.dtype
        self.clip_dim = clip_model.text_projection.shape[1]

        # frozen CLIP text components
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        for p in self.token_embedding.parameters():
            p.requires_grad_(False)
        for p in self.transformer.parameters():
            p.requires_grad_(False)
        for p in self.ln_final.parameters():
            p.requires_grad_(False)
        self.positional_embedding.requires_grad_(False)
        self.text_projection.requires_grad_(False)

        # causal attention mask (built explicitly, avoids open_clip internal API)
        ctx_len = self.positional_embedding.shape[0]  # 77
        mask = torch.empty(ctx_len, ctx_len)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        self.register_buffer('attn_mask', mask)

        # tokenize template, split into frozen prefix / suffix around the learnable slot
        tokenized = tokenizer([_TEMPLATE])  # (1, 77)
        with torch.no_grad():
            embedding = self.token_embedding(tokenized).type(dtype)  # (1, 77, ctx_dim)
        self.register_buffer('token_prefix', embedding[:, :1 + _N_CTX, :])           # SOS + "A photo of a"
        self.register_buffer('token_suffix', embedding[:, 1 + _N_CTX + _N_CLS_CTX:, :])  # "person." + EOS + pad
        self.register_buffer('tokenized_prompts', tokenized)
        self._dtype = dtype

        # learnable per-ID context
        cls_vectors = torch.empty(num_classes, _N_CLS_CTX, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)
        print(f'[CLIP-ID-Prompt] CoOp prompts: {num_classes} IDs x {_N_CLS_CTX} ctx x {ctx_dim}, '
              f'clip_dim={self.clip_dim}, CLIP text encoder FROZEN')

        # Option B: pose-conditioned prompt — per-image pose modulates the per-ID context
        self.pose_cond = pose_cond
        if pose_cond:
            _rng = torch.get_rng_state()   # preserve RNG so downstream module inits match exp341 (codex B Medium-1)
            self.pose_encoder = nn.Sequential(
                nn.Linear(pose_dim, ctx_dim), nn.ReLU(inplace=True),
                nn.Linear(ctx_dim, _N_CLS_CTX * ctx_dim))
            nn.init.zeros_(self.pose_encoder[-1].weight)   # start at 0-delta == exp341, then learn
            nn.init.zeros_(self.pose_encoder[-1].bias)
            torch.set_rng_state(_rng)
            print(f'[CLIP-ID-Prompt] POSE-COND (B): prompt context modulated by pose ({pose_dim}-d), zero-init, RNG-preserved')

    def forward(self, label, pose=None):
        """label: (B,) long -> (B, clip_dim) ID text prototypes. pose: (B, pose_dim) optional."""
        b = label.shape[0]
        cls_ctx = self.cls_ctx[label]                          # (B, n_cls_ctx, ctx_dim)
        if self.pose_cond and pose is not None:
            pose_delta = self.pose_encoder(pose.float()).view(b, _N_CLS_CTX, self.ctx_dim).type(self._dtype)
            cls_ctx = cls_ctx + pose_delta                     # pose-conditioned context (Option B)
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)
        prompts = torch.cat([prefix, cls_ctx, suffix], dim=1)  # (B, 77, ctx_dim)

        x = prompts + self.positional_embedding.type(self._dtype)
        # open_clip Transformer here is batch_first -> keep (B, seq, dim)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)
        tok = self.tokenized_prompts.expand(b, -1)
        x = x[torch.arange(b, device=x.device), tok.argmax(dim=-1)] @ self.text_projection
        return x.float()                                       # (B, clip_dim)


def supcon_i2t(image_feat, text_feat, labels, temperature=0.07):
    """Batch-level supervised contrastive (CLIP-ReID stage1). image_feat/text_feat L2-normed,
    same ordering (text_feat[i] is the prototype for labels[i]). Positives = same label."""
    image_feat = F.normalize(image_feat, dim=1)
    text_feat = F.normalize(text_feat, dim=1)
    logits = image_feat @ text_feat.t() / temperature         # (B, B)
    labels = labels.view(-1, 1)
    mask = labels.eq(labels.t()).float()                      # (B, B) positives
    logp = F.log_softmax(logits, dim=1)
    loss = -(mask * logp).sum(1) / mask.sum(1).clamp(min=1)
    return loss.mean()


class PoseGuidedPool(nn.Module):
    """Option A: LGPA-style pose-bias pooling → a pose-guided (occlusion-aware) global
    feature for the CLIP-ID-prompt to ALIGN. A learnable query attends the backbone tokens,
    additively biased by the person pose heatmap (de-emphasizes occluders/background).
    Pose guides WHAT the CLIP mechanism aligns (vs raw GAP global)."""
    def __init__(self, dim, pose_temp=1.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.k_proj = nn.Linear(dim, dim)
        self.pose_temp = float(pose_temp)

    def forward(self, featmap, pose_heatmap):
        # featmap (B,C,H,W); pose_heatmap (B,K,Hh,Ww)
        B, C, H, W = featmap.shape
        tokens = featmap.flatten(2).transpose(1, 2)              # (B, N, C)
        k = self.k_proj(tokens)                                  # (B, N, C)
        attn = (k @ self.query) / (C ** 0.5)                     # (B, N)
        pose = F.interpolate(pose_heatmap.float(), size=(H, W),
                             mode='bilinear', align_corners=False)
        pose_region = pose.amax(dim=1).flatten(1)                # (B, N) person visibility
        attn = attn + self.pose_temp * pose_region
        attn = F.softmax(attn, dim=1)                            # (B, N)
        pooled = torch.einsum('bn,bnc->bc', attn, tokens)        # (B, C) pose-guided feature
        return pooled


class PoseGuidedPartPool(nn.Module):
    """Option C: K pose-LOCALIZED part features. Each of K learnable queries pools the
    backbone tokens biased by ITS body part's keypoint heatmap (head / torso+arms / legs).
    Each part feature is then aligned to the per-ID prototype (pose-localized part-level
    CLIP alignment) — finer-grained than Option A's single global pose pooling."""
    PART_GROUPS = [[0, 1, 2, 3, 4],            # head: nose, eyes, ears
                   [5, 6, 7, 8, 9, 10],        # torso+arms: shoulders, elbows, wrists
                   [11, 12, 13, 14, 15, 16]]   # legs: hips, knees, ankles

    def __init__(self, dim, pose_temp=1.0):
        super().__init__()
        self.n_parts = len(self.PART_GROUPS)
        self.queries = nn.Parameter(torch.randn(self.n_parts, dim) * 0.02)
        self.k_proj = nn.Linear(dim, dim)
        self.pose_temp = float(pose_temp)

    def forward(self, featmap, pose_heatmap):
        # featmap (B,C,H,W); pose_heatmap (B,17,Hh,Ww) -> (B, n_parts, C)
        B, C, H, W = featmap.shape
        tokens = featmap.flatten(2).transpose(1, 2)             # (B, N, C)
        k = self.k_proj(tokens)                                 # (B, N, C)
        pose = F.interpolate(pose_heatmap.float(), size=(H, W),
                             mode='bilinear', align_corners=False)  # (B, 17, H, W)
        part_feats = []
        for i, grp in enumerate(self.PART_GROUPS):
            bias = pose[:, grp].amax(dim=1).flatten(1)          # (B, N) part-i visibility
            attn = (k @ self.queries[i]) / (C ** 0.5) + self.pose_temp * bias
            attn = F.softmax(attn, dim=1)                       # (B, N)
            part_feats.append(torch.einsum('bn,bnc->bc', attn, tokens))  # (B, C)
        return torch.stack(part_feats, dim=1)                   # (B, n_parts, C)


class PoseWeightedPool(nn.Module):
    """exp347: PARAMETER-FREE pose-visibility-weighted pooling. No learnable query/proj →
    the i2t/t2i alignment gradient flows DIRECTLY into the backbone (no pathway to absorb it,
    unlike Option A's PoseGuidedPool which had learnable params and absorbed → 57.6).
    Aligns the DE-OCCLUDED global (visible-person weighted) to the pure-ID prototype; the
    backbone is shaped to make visible-person features ID-discriminative. Descriptor stays raw GAP."""
    def __init__(self, pose_temp=4.0):
        super().__init__()
        self.pose_temp = float(pose_temp)

    def forward(self, featmap, pose_heatmap, invert=False):
        # featmap (B,C,H,W); pose_heatmap (B,17,Hh,Ww) -> (B, C) pooled global, NO params.
        # invert=True → weight the OCCLUDER (low-visibility) regions (exp348 occluder repulsion).
        B, C, H, W = featmap.shape
        tokens = featmap.flatten(2).transpose(1, 2)             # (B, N, C)
        pose = F.interpolate(pose_heatmap.float(), size=(H, W),
                             mode='bilinear', align_corners=False)
        vis = pose.amax(dim=1).flatten(1)                       # (B, N) person visibility
        w = ((-vis if invert else vis) * self.pose_temp).softmax(dim=1)  # (B, N), no params
        return torch.einsum('bn,bnc->bc', w, tokens)           # (B, C)


class CLIPVisualEncoder(nn.Module):
    """exp356 PC-MSC: frozen CLIP ViT visual encoder. Produces per-region (head/torso/legs =
    top/mid/bottom thirds of the 16x16 patch grid) dense-token features as the masked-semantic-
    completion TARGET. Input is the SOLIDER-normalized image (0.5/0.5); re-normalized to CLIP.
    Entirely frozen + @no_grad: never trains, never in the optimizer."""
    _MEAN = (0.48145466, 0.4578275, 0.40821073)
    _STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, clip_arch='ViT-L-14', clip_pretrained='openai'):
        super().__init__()
        clip_model, _, _ = open_clip.create_model_and_transforms(clip_arch, pretrained=clip_pretrained)
        self.visual = clip_model.visual
        for p in self.visual.parameters():
            p.requires_grad_(False)
        self.visual.eval()
        self.clip_dim = clip_model.text_projection.shape[1]
        self.register_buffer('mean', torch.tensor(self._MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(self._STD).view(1, 3, 1, 1))
        self._tok = {}
        self.visual.transformer.resblocks[-1].register_forward_hook(
            lambda m, i, o: self._tok.__setitem__('t', o))
        print(f'[PC-MSC] CLIP visual encoder FROZEN, clip_dim={self.clip_dim}')

    @torch.no_grad()
    def part_targets(self, img):
        """img: (B,3,H,W) SOLIDER-normalized (0.5/0.5). Returns (B,3,clip_dim) L2-normed
        per-region (head/torso/legs) frozen CLIP features as the completion target."""
        wdtype = self.visual.conv1.weight.dtype
        x = img.float() * 0.5 + 0.5                          # un-norm SOLIDER (mean=std=0.5) -> [0,1]
        x = (x - self.mean) / self.std                       # CLIP norm
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        self._tok.clear()
        _ = self.visual(x.to(wdtype))                        # triggers hook; pooled output discarded
        t = self._tok['t']
        if t.shape[0] != x.shape[0]:                         # (seq, B, dim) -> (B, seq, dim)
            t = t.permute(1, 0, 2)
        patch = self.visual.ln_post(t[:, 1:])                # drop CLS -> (B, 256, dim)
        if getattr(self.visual, 'proj', None) is not None:
            patch = patch @ self.visual.proj                 # (B, 256, clip_dim)
        B = patch.shape[0]
        grid = patch.reshape(B, 16, 16, -1).float()
        regions = torch.stack([grid[:, :5].mean((1, 2)),     # head  (top 5 rows)
                               grid[:, 5:11].mean((1, 2)),    # torso (mid 6 rows)
                               grid[:, 11:].mean((1, 2))], dim=1)   # legs (bottom 5 rows)
        return F.normalize(regions, dim=-1)                  # (B, 3, clip_dim)
