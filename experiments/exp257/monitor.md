# exp257 ArcFace + Label Smoothing on Small GCN512 + 2-stage PSG

## 变体
- exp257: ArcFace + Label Smoothing (远程 5060Ti)
- exp257b: Label Smoothing only (本地 3090)

对照: exp255 (softmax, no LS): 73.2/83.3, MaxSim 73.5/83.8

## 检查点

### [05:24] 启动

exp257b (本地) ep1, healthy. 8856 MiB.
exp257 (远程) 启动成功。

### [13:43/05:44] 检查点 #2 (cron) — exp257 ep3, exp257b ep5

**exp257 (ArcFace+LS)**: ep3, id_global=15.94 (ArcFace scale=30 放大, 正常), Acc=0.000 (ArcFace 早期正常 — margin 使正确类 logit 为负)。需 ep10+ 才能判断。
**exp257b (LS only)**: ep5, id_global=6.509, Acc=0.096. 正常早期学习。

### [14:13/06:14] 检查点 #3 (cron) — exp257 ep7, exp257b ep10 eval imminent

**exp257**: ep7, id_global 15.94→13.57 (在学习), Acc=0.005 (ArcFace slow start).
**exp257b**: ep10 iter 200, eval ~1 min.

### [06:15] ⚠️ exp257b ep10 = 51.1/62.2 — LS 负面!

vs exp255 54.7/65.7 = **-3.6/-3.5!** Label Smoothing 抹平了 GCN512 早期优势。继续观察。

### [14:45] ⭐ exp257 (ArcFace+LS) ep10 = 54.5/69.1 — R1 +3.4!

vs exp255 54.7/65.7 = -0.2/**+3.4!** ArcFace R1 超强!
exp257b (LS only) ep16 training.

### [15:13/07:14] 检查点 #5 (cron) — exp257 ep14, exp257b ep20=60.4/72.0

**exp257b ep20**: 60.4/72.0 (vs exp255 62.2/74.3 = -1.8/-2.3). LS gap 缩小。

### [16:10/08:04] 检查点 #7 — exp257 ep20=60.0/74.9, exp257b ep30=65.1/76.9

**exp257 ep20**: 60.0/74.9 (vs exp255 62.2/74.3 = -2.2/**+0.6**). mAP 落后但 R1 仍领先!
**exp257b ep30**: 65.1/76.9 (vs exp255 67.0/77.4 = -1.9/-0.5). LS gap 继续缩小。

| Epoch | exp257 (AF+LS) | exp257b (LS) | exp255 (baseline) |
|-------|---------------|-------------|-------------------|
| 10 | 54.5/**69.1** | 51.1/62.2 | 54.7/65.7 |
| 20 | 60.0/**74.9** | 60.4/72.0 | 62.2/74.3 |
| 30 | — | 65.1/76.9 | 67.0/77.4 |

### [17:13/09:13] 检查点 #9 — exp257 ep28, exp257b ep40=68.0/79.2

**exp257b ep40**: 68.0/79.2 (vs exp255 70.2/81.2 = -2.2/-2.0). LS gap 稳定 ~-2.0, 不再缩小。
LS 在 Small GCN512 上确认负面 (~-2 mAP)。

### [17:34/09:44] 检查点 #10 — exp257 ep30=60.1/75.7 ⚠️, exp257b ep49

**exp257 ep30**: 60.1/75.7 (vs exp255 67.0/77.4 = **-6.9/-1.7**). ArcFace mAP 严重落后!
mAP vs R1 不对称: mAP -6.9 但 R1 只 -1.7。ArcFace 收敛慢但 top-1 能力强。
**问题**: margin=0.35 可能太大, 或 ArcFace+LS 双重正则太强。

### [18:13/10:13] 检查点 #11 (cron) — exp257 ep35, exp257b ep50=69.8/80.3

**exp257b ep50**: 69.8/80.3 (vs exp255 71.3/82.0 = -1.5/-1.7). LS gap 微缩。

### [18:58/10:46] 检查点 #13 — exp257 ep40=57.1/75.3 ⚠️⚠️, exp257b ep60=70.3/81.1

**exp257 ep40**: 57.1/75.3 — **mAP 从 ep30 60.1 下降到 57.1!** ArcFace mAP 崩溃。
vs exp255 70.2/81.2 = **-13.1/-5.9**. ArcFace margin=0.35 太大，mAP 恶化。
**应考虑止损** — ArcFace 对 mAP 严重有害。

**exp257b ep60**: 70.3/81.1 (vs exp255 71.6/82.1 = -1.3/-1.0).

### [19:43/11:44] 检查点 #14 (cron) — exp257 ep46, exp257b ep70=70.9/81.3

**exp257b ep70**: 70.9/81.3 (vs exp255 71.9/81.8 = **-1.0/-0.5**). LS gap 继续缩小!
趋势: ep40 -2.2, ep50 -1.5, ep60 -1.3, **ep70 -1.0**。可能 final ~-0.5.

### [20:22/12:34] 检查点 #16 — exp257 ep50=59.1/76.5, exp257b ep80=71.5/81.7

**exp257 ep50**: 59.1/76.5 (recovering from ep40 57.1, but vs exp255 -12.2). ArcFace 太慢。
**exp257b ep80**: 71.5/81.7 (vs exp255 72.7/82.5 = **-1.2/-0.8**). LS gap 继续缩小!

| ep | exp257b (LS) vs exp255 delta mAP |
|----|-------------------------------|
| 10 | -3.6 |
| 20 | -1.8 |
| 30 | -1.9 |
| 40 | -2.2 |
| 50 | -1.5 |
| 60 | -1.3 |
| 70 | -1.0 |
| **80** | **-1.2** |

LS 稳定在 -1.0~-1.3，final 可能 ~72.0 (vs exp255 73.2 = -1.2)。LS 确认略负。
