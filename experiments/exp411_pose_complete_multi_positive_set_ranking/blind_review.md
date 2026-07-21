# exp411 PCMPSR 独立代码盲审

## 审查范围

- 冻结design/protocol/config；
- `loss/pose_clip_multi_positive_set.py`、`loss/make_loss.py`；
- `processor/processor.py`、`model/make_model.py`、`config/defaults.py`；
- fresh cache builder与synthetic/real-batch合同计划；
- default-off、等支持、owner/control、cache provenance、互斥、FP32/AMP、eval边界及Torch 1.13/Python 3.8。

## 首轮结论

独立智能体盲审=`0 BLOCKER / 2 HIGH`，未修改文件、未使用GPU。

### HIGH-1：pose-invisible owner未完全屏蔽

首版correct只mask `clip_valid`；当可见候选cosine为负时，visibility为0的候选可能以0分胜出，违反冻结的
pose-visible owner合同。修复为：正常owner候选严格满足`clip_valid & visibility>0`；只有整槽无合法候选时才进入
显式、确定性的pose-first fallback，并单独报告fallback fraction。synthetic合同新增负向断言：只要合法候选存在，
最终owner必为pose-visible且CLIP-valid。

### HIGH-2：真实default-off与CUDA/AMP梯度门尚未覆盖

首版synthetic只证明set/support/owner/loss及final feature梯度，不能宣称真实D0 model state/RNG/forward/loss exact，
也不能代替Stage-3/backbone与optimizer update。按冻结协议不增加多轮测试：fresh cache完成后只运行一次真实PK64
MMPOSE-ABU CUDA/AMP合同，同时覆盖D0-vs-default-off同seed state/RNG/forward/loss exact、correct与controls active、
FP32 listwise、final descriptor/Stage-3/backbone finite nonzero gradient和一次真实optimizer update。该合同PASS前student
保持NO-START。

## 当前闭环状态

HIGH-1代码与负向contract已修；HIGH-2的真实合同边界已冻结，等待fresh cache后一次执行。聚焦复审需确认修复代码
无新BLOCKER/HIGH，但任何复审结论都不能把尚未执行的真实CUDA合同提前写成PASS。

首轮聚焦复审指出真实合同还需隔离PCMPSR本身梯度，不能用CE+PCMPSR+pose总梯度代替；同时default-off RNG必须
覆盖构造期及forward后的Python、NumPy、Torch CPU和全部CUDA状态。脚本现已按此补充：未detach set loss分别检查
descriptor/Stage-3/backbone，combined loss只负责生产AMP update；两套config在构造后与同seed forward后比较四类
RNG。fallback也统一为文档所述pose-first。上述仍属于待执行合同，不提前记为PASS。

## 最终聚焦复审

最终结论=`0 BLOCKER / 0 HIGH`。复审确认：正常owner候选与fallback语义一致；default-off脚本分别比较同seed构造后
及共同forward seed前后的Python、NumPy、Torch CPU、全部CUDA RNG，并比较state/forward/loss exact；未detach的
isolated set loss分别检查descriptor、Stage-3与backbone梯度，combined loss只负责native GradScaler backoff和真实
Stage-3 update。所用API兼容Torch 1.13。复审未修改文件、未使用GPU；真实合同仍须fresh cache后实际执行。
