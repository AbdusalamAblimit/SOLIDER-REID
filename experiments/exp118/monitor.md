# exp118 监控

## 实验信息
- 方法: PAA+ROA(p=0.7)+VCGA 组合
- 类型: 组合验证
- 运行位置: 本地 3090
- 主配置: `configs/occluded_duke/pose_psg_gcn_paa_roa_vcga.yml`
- 核心变量: `POSE_VCGA = True`（在 PAA+ROA 基础上）
- 对照组: `exp085-eq`（PAA+ROA p=0.7 = 62.6% / 75.3%）

## 启动记录

### [2026-03-20 09:30] 启动确认
- 训练正常启动
- 当前判断: 继续
