# exp411 PCMPSR 冻结协议

## 不变量

- official数据只读`/mnt1/afrdata`，pose只读`/mnt1/afrderived`；
- 固定MMPOSE-ABU、Swin-Tiny、batch64、P×K=16×4、seed1234、e120、fresh OUTPUT_DIR、不续训；
- learned CE、D0 pose loss、optimizer、schedule、augmentation、global descriptor与eval不变；
- exp411 correct首臂之外无GPU并行任务；PICRD/PCHM/PC²P及所有SEALED编号禁止重跑；
- fresh exp411 cache不得读取或复制exp408/409/410运行asset，builder从official RGB与冻结pose重新编码；
- 运行中冻结source/config/cache/参数，不按中间性能早停。

## 冻结机制

1. 每个anchor按类内位置q，为每个batch身份排除同位置样本，形成相同大小的3图支持集；
2. 每槽owner仅在该支持集内按`pose visibility × CLIP-to-set-slot-consensus cosine`离散argmax；
3. 身份集合距离等权包含3个原support距离与5个slot-owner距离；
4. `softplus(logmeanexp(D_pos-D_neg_set))`替换原batch-hard triplet；
5. owner选择stop-gradient，CLIP/pose不决定连续loss尺度，测试期完全删除。

## 一次性执行顺序

1. 冻结design/protocol/文献代码审计；
2. 实现default-off开关、fresh cache schema/loader、owner与set loss、processor接线；
3. 必要语法/shape/default-exact/真实PK64 CUDA-AMP合同；
4. 一次独立智能体代码盲审，只修BLOCKER/HIGH，闭环0B/0H；
5. fresh构建全15,618图cache，核验覆盖、SHA、五槽valid/norm/provenance；
6. 若真实PK64机制/梯度门PASS，立即fresh启动唯一correct e120；
7. e10/20/.../120记录PCMPSR与sealed clean D0同epochmAP/R1，最终只以自然e120裁决；
8. 性能FAIL即封板并进入exp412新对象；性能GO才串行zero-owner和wrong-RGB matched controls。

## 裁决门

- 机械门：支持集/owner合法、default-off exact、control active、CUDA/AMP真实update；
- 性能门：自然e120 raw mAP `>57.5587756578`且R1 `>67.6923076923`；
- 归因门：correct严格胜zero-owner与wrong-RGB matched e120；
- 任一runtime/validity失败只封板该fresh执行路径，不伪造结果、不续训；科学FAIL不得修改旧机制救臂。

## correct性能GO后的matched-control执行

correct已自然e120=`58.8/70.1/82.1/85.8`并永久封板。归因阶段以
`matched_controls.md`为冻结合同：先fresh `zero_owner`，其唯一变量是从集合距离删除五个owner multiplicity；
自然封板后再fresh `wrong_rgb`，其唯一变量是owner选择所用CLIP槽按固定different-PID shift=4轮换。两臂均共享
三图support、全身份set ranking、cache、pose、student配置和自然e120协议。correct必须在mAP与R1上严格胜二者，
才允许形成pose+CLIP归因。
