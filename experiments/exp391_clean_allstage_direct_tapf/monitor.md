# exp391 监控：官方 clean 全阶段独立直预测 TAPF

## 当前状态

- 状态：`PREFLIGHT / NO-START`；exp390已封板，当前只进入Phase A=`H2-M`实现与门禁；
- H2-M唯一变量：在保持exp389两个direct anchor、early/late=`6/2` consumer和全部recipe不变时，
  将总pose objective从`0.1×sum(L_early,L_late)`改为`0.1×mean(L_early,L_late)`；
- 正式output=`log/occluded_duke/exp391_clean_swin_tiny_h2m_s1234`，门禁完成前不得创建或启动；
- Phase B/C继续保持`NO-START`，只有前一阶段按design.md通过才允许实现或训练。

## 不变量

- official clean runtime、teacher、exp386 train-only artifact、batch64、seed1234、120 epoch、SGD、
  lr0.0008、semantic weight0.2、增强/sampler/eval10/checkpoint120固定；
- 不并行、不续训、不重复、不挑best、不按中间性能停止；不修改运行中代码/config；
- query/gallery严格RGB-only；correct/shuffle/None/exploding external pose必须exact；
- 每个阶段先完成state/init/RNG/optimizer、route/gradient、AMP overflow、strict load、consumer path与
  参数/效率门禁，再决定是否fresh正式启动；
- 保护用户工作树，只显式暂存目标文件，禁止`git add -A`。

## Phase A 本地实现边界

- 默认新增`MODEL.TAPF.POSE_LOSS_REDUCTION='sum'`，因此既有D0、HT0和config-off路径保持默认语义；
  只有H2-M config显式设为`mean`；
- `CleanTapfHt0`只在合并early/late pose loss时执行`sum`或乘`0.5`，两个单独pose loss、state keys、
  anchor/PSG构造顺序、forward route和optimizer参数均不改变；非法reduction严格抛错；
- H2-M config相对exp389 formal HT0 config的文本diff只有
  `POSE_LOSS_REDUCTION: mean`与独立`OUTPUT_DIR`；
- 本地`uv`环境的四个修改Python文件`py_compile` PASS；该环境未安装`cv2/torchvision`，因此首次
  unittest在导入阶段退出、未执行任何case。它不计作单元测试结果；必须在远端canonical
  `mmpose-abu`环境完整重跑后才可继续门禁。
