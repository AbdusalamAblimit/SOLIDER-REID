# exp098 PKP 代码审查

## 审查通过 ✅

- ✅ Shape: heatmap (17,96,32) → resize (17,384,128) → Conv2d → (96,96,32) → flatten (3072,96) = x shape
- ✅ Zero-init: Conv2d weight+bias 全零 → 初始 pose_tokens=0 → identity start
- ✅ Gradient: pose_prompt_embed 是正常 nn.Conv2d，梯度正常流过
- ✅ Config: POSE_PKP=False 时无行为变化
- ✅ Memory: +300-400MB（临时 heatmap resize 开销），可接受
- ✅ 向后兼容：所有现有实验不受影响
