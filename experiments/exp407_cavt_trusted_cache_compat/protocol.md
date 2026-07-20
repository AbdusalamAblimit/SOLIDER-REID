# exp407 CAVT P0B协议

## 冻结身份

- preflight：`exp407-p0b-preflight-v1`
- output：`/home/afr/reid-clean/audits/exp407-p0b-preflight-v1`
- formal：`exp407-p0b-iso-teacher-v1`
- formal output：`/home/afr/reid-clean/audits/exp407-p0b-iso-teacher-v1`
- asset：`/home/afr/reid-clean/assets/exp407-p0b-preflight-v1`
- runtime：`/usr/local/anaconda3/envs/mmpose-abu/bin/python`
- official：只读`/mnt1/afrdata`
- pose：只读`/mnt1/afrderived/exp386_occluded_duke_vitpose_huge_train`

exp407不得读取exp405/exp406的output、cache、pair map、MAD、receipt或path mapping。core和teacher复制冻结字节；
donor reserve、20对diagnostic、删除比例、wrong-mask、阈值、batch和formal科学门沿用既定合同。donor排序保留历史
冻结salt `exp406-donor`；该常量只决定official输入的确定性顺序，不读取或授权任何exp406运行产物。

## Cache发布

runner先以exclusive temporary写入、flush、fsync，再用`weights_only=False`回读schema；该文件由同一进程刚写出，
回读发生在rename发布之前，不接受任何外部cache路径。schema不符立即失败，成功后原子rename、fsync目录并记录SHA。

## 最小验证与执行

1. 固定MMPOSE-ABU中运行一次targeted roundtrip，两次fresh输出必须byte-exact；
2. 一次独立聚焦盲审，BLOCKER/HIGH必须清零；
3. 核对fresh output/started/failure/asset不存在、输入SHA与GPU独占；
4. 启动唯一preflight，自然完成，不按中间指标终止；
5. PASS后另建fresh formal manifest，formal不得复用preflight feature、scale、pair或cache。

任何runtime/validity失败均写receipt并封板当前execution；不得同编号重跑或补结果。
