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
    def __init__(self, num_classes, clip_arch='ViT-B-32', clip_pretrained='openai'):
        super().__init__()
        clip_model, _, _ = open_clip.create_model_and_transforms(clip_arch, pretrained=clip_pretrained)
        tokenizer = open_clip.get_tokenizer(clip_arch)
        ctx_dim = clip_model.token_embedding.weight.shape[1]
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

    def forward(self, label):
        """label: (B,) long -> (B, clip_dim) ID text prototypes."""
        b = label.shape[0]
        cls_ctx = self.cls_ctx[label]                          # (B, n_cls_ctx, ctx_dim)
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
