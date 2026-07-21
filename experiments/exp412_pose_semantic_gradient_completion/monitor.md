# exp412 PSGC 监控记录

## 2026-07-22：设计冻结前状态

exp411 wrong-RGB 仍为远端唯一 4090 任务，PSGC 尚未传输、未生成 text asset、未执行 CUDA 合同或正式训练。
当前仅建立新问题与机制设计：以 sealed zero-owner 集合排序为宿主，永久删除失败的 owner multiplicity，只让
pose visibility 与身份无关 CLIP visible-vs-occluded 文本差值组织同 PID×槽的守恒路由系数预算。下一步为独立
盲审设计和最小实现；wrong-RGB 自然 e120 封板前严格 GPU NO-START。

## 2026-07-22：实现盲审通过与一次性 text asset 冻结

PSGC 设计首轮独立盲审的 `0B/3H` 已全部闭环：router 固定在 `norm3` 后、avgpool 前并强制路由场与 feature
同 dtype/device；新增 q-only 控制；“梯度守恒”表述收紧为“路由系数预算守恒”。实现终审为 `0B/0H`，数学
Pareto 方向、四 control 共享候选、K=4 系数预算、AMP forward-exact 接点、processor、eval/default-off 与 asset
loader 均无致命问题。

已从运行中 exp411 formal 只读 clone 建立独立 fresh formal=
`/home/afr/SOLIDER-REID-exp412-psgc-formal-v1`，实现提交 HEAD=
`a205599a1389aa08852a291fab1ae5aa05722263`。未修改 exp411 运行源。固定 MMPOSE-ABU 在
`CUDA_VISIBLE_DEVICES=''` 下仅用 CPU 一次性生成 5×2×768 visible/occluded text axes；asset=
`/home/afr/reid-clean/assets/exp412-psgc-text-v1/text_axes.npz`，SHA=
`3b5fd399350a522459cef090bff72d76c97ff483921d84aae4990eda16727bf3`，prompt-spec/builder SHA=
`2958675be9a258ab098363588bf90dce752bc39d51efdf10a947a07d92ba3c2c`/
`2daf7b25a1814100c02525adbc1333aa19c8e072575ec9ec1465a17890ae0bb2`，绑定 CLIP checkpoint SHA=
`9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e`。builder 自然退出，期间4090仍只有
wrong-RGB PID；当前 PSGC 继续 GPU NO-START，等 exp411 自然封板后只跑一次真实 PK64 合同。
