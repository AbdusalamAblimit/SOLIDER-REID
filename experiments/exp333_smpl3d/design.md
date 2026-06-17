# 实验 exp333: SMPL-3D 外部信息源辅助分支（occluded ReID）

## 动机
- 334 篇全量文献 mine（`experiments/lit_review_occluded_2025_2026.md`）结论：近年 occluded-ReID 机制 18/23 塌回我们已实测判死的家族（可见性加权 / 特征补全 / CVK / FM-import / backdoor / re-rank）→ 强力反证"吸收陷阱"（任何"输出是单图像素的可微函数、且与 ReID 度量联合优化"的机制都被 backbone 内化）。
- **唯一真正逃出吸收陷阱的方向 = 引入外部信息源**。SMPL 3D 人体重建提供**与像素无关的身体几何**：身体被遮时，参数化人体先验（ROMP，在 3D mocap 上训练、与 ReID 度量无关）仍把整具身体补全出来。backbone 无法复现/吸收它，因为它不是 backbone 特征的函数。
- 用户判断：SMPL 作为新信息源应当涨点。

## Stage A 硬门（已通过，2026-06-18，lab-3090-d）
ROMP 在 Occ-Duke `bounding_box_test` 30 ids × ≤8 图（跨摄像头）：
- 检测率 **84.2%**（202/240）；16% 重遮挡无 SMPL → 分支需 missing fallback。
- **β(体型,10d) 跨摄像头 NN 身份准确率 = 0.153 vs 随机 0.033 = 4.6×**；sep_ratio 1.156。
- body_pose 0.272（8.2×，但有同 tracklet 姿态相关嫌疑，不作主特征）；β+pose 0.292。
- 脚本 `scripts/smpl_fit_validation.py`，原始 betas 存 `/tmp/smpl_valA/`。
- **结论**：β 在低分辨率遮挡监控图上有真实跨摄像头身份信号 → 有资格往下建分支。

## 核心假设（一句话）
把外部 ROMP 估计的 **SMPL 体型嵌入**（来自 β）与外观特征融合，能提升遮挡 ReID——尤其在外观被破坏但体型可由人体先验恢复的重遮挡 query 上；因为 β 是 backbone 学不到的外部正交身份线索。

## 为何逃出吸收陷阱
SMPL 参数由**冻结的外部估计器**（ROMP，3D 数据训练）算出，不是 ReID backbone 特征的可微函数 → backbone 无法内化复现。这是 9 连负里所有死法都不具备的性质（它们的机制输出都是单图像素的可微函数）。

## 技术方案
### Stage B：离线缓存 SMPL（preprocessing，非训练）
- 对 train/query/gallery 全量图跑估计器，按图名缓存 `β(10)`、`body_pose(69)`、`global_orient(3)`、`cam(3)`、`center_conf`、`valid_flag`、人数。
- **目标人选择**（关键，避开 person-occluder）：多人时选投影中心**最靠近图像中心**的检测（ReID crop 以目标为中心）；同时把第 2 中心人的 β 也存下（inter-person 遮挡物表征红利，备用）。
- 缓存成单个 npz（dict: filename → 向量），dataloader 按图名 O(1) 取。

### Stage C：3D 辅助分支 + 训练（需 design 双审后才训）
- **Baseline = 弱 TransReID ViT-base**（有 headroom ~53-59 mAP，合法验证场；强 SOLIDER 栈会压没增益）。
- **3D 分支**：输入 `[β(10), valid_flag]`（主），ablation 再加 body_pose。小 MLP（如 11→256→256）+ BN → L2 归一 3D embedding `e_3d`（256d）。**缺检测的 16% 用一个 learnable missing-embedding**。
- **双分支设计**（隔离 3D 贡献、不破坏外观 baseline）：外观分支原样（ID+triplet）；3D 分支独立 ID+triplet；推理时 concat `[f_app(768) || e_3d(256)]`。
- **单变量**：同 config/seed，只开关 3D 分支。
- 3D 分支**不回传到 SMPL 参数**（预计算输入，detach）。

## 预期结果
- 假设成立：弱 baseline +1~3 mAP，重遮挡子集涨更多。
- **诚实风险**：外观（~53 mAP）远强于 β 单独（4.6× chance）。β 要涨点必须**正交**（救外观失败的重遮挡case）。若失败最可能：(a) 低分辨率 β 估计噪声大、在外观尚存时加不上；(b) 外观 backbone 已隐式编码体型。但 Stage A 证 β 有独立信号 + 它是外部输入（抗吸收）→ 期望为正。
- 评测：整体 mAP/R1 + **重遮挡 query 子集**（按 SMPL center_conf 低 / 多人 分层）。

## 对照组
- baseline = 同 config/seed 的弱 TransReID，无 3D 分支。
- 消融：β-only vs β+pose；fusion concat vs 仅 3D-head 加权。

## 实现说明（最终方案，2026-06-18）
- **自包含训练脚本** `scripts/exp333_train_smpl3d.py`，复用仓库精确机制（RandomIdentitySampler / 同 transforms / `make_loss`（soft-margin triplet + CE）/ `make_optimizer`（SGD）/ `create_scheduler`（cosine）/ `R1_mAP_eval`）。**不碰共享 processor.py**（避免破坏其他实验）。
- **单变量 A/B**：`--use_smpl` 开关。off=纯外观 baseline（control）；on=同外观分支 + 3D 分支。其余（backbone/采样/loss/优化器/调度/seed）逐字相同。
- **ViT backbone 直建**（重要）：本仓库 `build_transformer` 把 `base(x)` 解包成 `(global_feat, featmaps)`，但 ViT 的 `TransReID.forward` 只返回单个 `x[:,0]`（768d）→ `build_transformer` 的 ViT 路径实际是坏的（swin-only，所以 exp329 当年用的是独立 TransReID 子模块）。故我**直接实例化** `vit_base_patch16_224_TransReID` + 标准 TransReID 头（BN bottleneck + 线性分类器），SIE off（camera=0/view=0）。pretrain=`pretrained/jx_vit_base_p16_224-80ecf9dd.pth`，`load_param(hw_ratio=1)`（方形 14×14 pretrain 插值到 16×8）。
- **3D 分支**：`smpl_mlp`(in_dim→256→BN→ReLU→256) → `bn3d` → `classifier3d`；缺检测图(valid=0)用 learnable `missing` 向量替换 MLP 输出。in_dim=10(beta) 或 79(beta+pose)。3D 分支独立 CE+triplet（`make_loss` 同一函数），总损失 = loss_app + w3d·loss_3d。
- **SMPL 输入预处理**：用**训练集 valid 样本**的 mean/std 对 beta(/pose) z-normalize；缺检测→zeros + valid=0。
- **测试融合**：外观特征(before-BN) 与 3D 嵌入(after-BN) **各自 L2 归一**后按 `alpha` 加权 concat（alpha∈{0,0.5,1,1.5,2} 扫，alpha=0 即纯外观但来自联合训练模型）。headline = 最佳 alpha 的 mAP vs baseline mAP。
- **缓存**：`cache/smpl/occduke_{train,query,gallery}.npz`，train valid≈93.3%。
- **诚实标注**：测试期 alpha 是融合超参（test-time），模型贡献 = "加 3D 分支" 整体 vs baseline；不把 alpha 调参算训练端增益。无 flip-test（两臂一致，公平）。

## 机器
- lab-3090-d（3090 全空，658G 盘，simple_romp 已装 `--user`，模型在 `/root/.romp/`）。两臂同机顺序跑（同设备公平 A/B），或并行到 lab-4090/hyy（需拷 cache）。
