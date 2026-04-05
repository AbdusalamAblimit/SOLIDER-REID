# exp245 Claude Review: LGPA-D on Small

## 审查范围
Config-only 改动: Tiny→Small backbone, LR 调整, TEST batch 降低。

## 验证
| 检查项 | 状态 |
|--------|------|
| Backbone 切换正确 | PASS (swin_small, pretrained/swin_small.pth) |
| LR 0.0004 | PASS (Small 标准) |
| TEST.IMS_PER_BATCH 128 | PASS (防 OOM) |
| LGPA-D 不变 | PASS |
| 单变量 (仅 backbone) | PASS |
| OA-SD 兼容 | PASS |

## 结论

审查通过。纯 config 改动, 无代码变更。
