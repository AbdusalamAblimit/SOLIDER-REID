# exp204 审查

## 审查范围
- 配置组合: SupCon + PLBOA + ROA 在 base arch (STD-PR) 上
- 无新代码修改，纯配置组合

## 检查项

1. **PLBOA + ROA 兼容性**: PLBOA 在 pose_dataset.py 的 branching 前应用(line 174-179)，ROA 在 standard path 内(line 225-238)。两者可共存——PLBOA 先 paste 下半身遮挡，ROA 再 paste VOC 物体。
2. **parallel_aug + ROA**: 在 parallel_aug mode 中，view_roa 已经使用了 ROA (line 209-215)。如果同时开 ROA_PROB=0.5 和 parallel_aug，ROA 会在 view_roa 中使用。但 exp204 不用 parallel_aug (远程 16GB)。
3. **配置安全**: 所有参数已有默认值，不影响其他实验。
4. **VOC 数据**: 远程已确认有 VOCdevkit。

## 结论

审查通过。纯配置组合，无风险。
