# exp245: LGPA-Detach on Swin-Small (泛化性验证)

## 动机
exp244 LGPA-D 在 Tiny 上 +2.1 mAP，是首个 final 正向的 Part branch。
需要在 Small 上验证泛化性 (PPA 在 Small 上灾难性失败 -9.7)。

## 核心假设
LGPA-D 的 detached 设计应该在 Small 上也能正向 (不像 PPA non-detach 那样干扰)。

## 技术方案
与 exp244 完全相同，仅换 backbone:
- Swin-Small + PSG + LGPA-D + OA-SD + PLBOA(0.7)
- LR 0.0004 (Small 标准)
- TEST.IMS_PER_BATCH 128 (防 OOM)

## 对照组
- exp206r (Small GCN+OA-SD): 70.6/82.6
- exp242 (Small PPA+GCN non-detach): 60.9/73.4 (灾难性失败)
