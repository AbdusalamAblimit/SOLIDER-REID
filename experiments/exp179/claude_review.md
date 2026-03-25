# Claude Broad Review: exp179 SupCon on base architecture (Opus 4.6)

## 审查通过

### a. design.md
清晰的消融实验：SupCon T=0.05 on exp166 base（PSG@Stage3 only, no PAPE, no multi-stage PSG）。测试 SupCon 是否独立于架构有效。

### b. 代码
无新代码。使用已有 SupCon 实现（loss/supcon_loss.py + make_loss.py:160-171）。

### c. 配置
Base config `pose_psg_stdpr_pertoken_plboa.yml`:
- No POSE_PATCH_EMBED → PAPE disabled (default False)
- No POSE_PSG_STAGES → Stage 3 only (default [-1])
- POSE_STR_PER_TOKEN: True → list return (7 elements)
- POSE_EVIDENTIAL: not set → SupCon branch reachable

CLI overrides: POSE_STR_SUPCON=True, POSE_STR_SUPCON_TEMP=0.05

### d. defaults.py
POSE_STR_SUPCON=False default safe. POSE_STR_SUPCON_TEMP=0.07 default safe.

### e. Loss integration
- SupCon condition met: True AND list AND len>3
- Replaces per-token CE with contrastive on feat[1:]
- Global CE on score[0] preserved
- Per-token triplet preserved (separate path)
- 'supcon' logged

### f. 单变量
vs exp166: only POSE_STR_SUPCON and POSE_STR_SUPCON_TEMP added.
All other settings (PSG, STD-PR, PLBOA, per-token, eval) identical.

零 issue。
