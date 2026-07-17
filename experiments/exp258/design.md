# exp258: ArcFace only (无 LS), margin=0.2, Small GCN512+2stage

## 动机
exp257 ArcFace+LS margin=0.35 mAP 崩溃。分离变量: 去掉 LS, 降低 margin 到 0.2。

## 变体
- exp258: ArcFace margin=0.2, 无 LS (本地)
- exp258b: GCN 3-layer hidden=512 (远程, 探索更深 GCN)
