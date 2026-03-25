# 实验 exp185: SupCon on STD-PR pooled mode (no per-token)

## 动机
- SupCon 在 per-token mode 效果强 (+1.0/+1.6 vs CE)
- 问题：SupCon 的增益是否依赖 per-token diversity structure？
- 如果在 pooled mode 也有效 → SupCon 不需要 per-token
- 如果无效 → SupCon 需要 per-token 的 diversity 结构来发挥

## 技术方案
- STD-PR + PLBOA 但 PER_TOKEN=False (pooled mode)
- SupCon T=0.05 on pooled feature (feat = [global, pooled_part], len=2)
- 代码已修改 len>1 条件，pooled mode 可以走 SupCon

## 对照组
- exp176 (SupCon + per-token): 64.1/75.5
- exp166 (CE + per-token): 63.1/73.9
