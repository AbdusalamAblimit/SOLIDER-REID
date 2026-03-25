# 实验 exp183: SupCon T=0.05 on base arch without PLBOA

## 动机
消融矩阵最后一个空格：
- exp166r: CE, no PLBOA, base = 60.3/72.8
- exp166: CE, PLBOA, full = 63.1/73.9
- exp179: SupCon, PLBOA, base = 64.2/74.9
- exp181: SupCon, no PLBOA, full = 59.8/70.4
- **exp183: SupCon, no PLBOA, base = ?**

## 技术方案
- Base config (PSG@Stage3, per-token, no PAPE, no multi-stage)
- SupCon T=0.05
- No PLBOA

## 对照组
- exp166r (CE, no PLBOA, base): 60.3/72.8
- exp179 (SupCon, PLBOA, base): 64.2/74.9
