# 实验日志 Git 归档

## 归档快照

- 归档日期：2026-07-25
- 来源主机：`lab4090`
- 本地仓库：`/Users/abdslm/Desktop/SOLIDER-REID`
- 权威运行目录原路径归档：632 个
- 全 `/home/afr` 训练/测试日志原路径：4,276 条
- 去重后的训练/测试日志内容：867 份，全部有Git归档副本
- 实际归档文件副本：1,153 个
- 已归档总字节数：124,126,447
- 远端与本地逐文件大小/SHA256：`PASS`
- 敏感信息模式扫描：0 命中

本目录只保存训练、测试、runner、preflight、formal receipt、PID 与小型 JSON 证据。实验源码、设计、监控、
结果和决策文档继续保存在各自的 `experiments/exp*` 目录，并与本快照一起进入 Git。

## 来源映射

| 归档目录 | 远端只读来源 |
|---|---|
| `lab4090/legacy_solider_runtime/` | `/home/afr/SOLIDER-REID/log/` |
| `lab4090/root_train_logs/` | `/home/afr/train-logs/` |
| `lab4090/pose_logs/` | `/home/afr/pose-logs/` |
| `lab4090/reid_clean/audits/` | `/home/afr/reid-clean/audits/` |
| `lab4090/reid_clean/formal_evidence/` | `/home/afr/reid-clean/formal/` |
| `lab4090/reid_clean/runner_logs/` | `/home/afr/reid-clean/train-logs/` |
| `lab4090/reid_clean/training_outputs/` | `/home/afr/reid-clean/logs/` |
| `lab4090/reid_clean/preflight_exp388/` | `/home/afr/reid-clean/preflight-exp388/` |
| `lab4090/reid_clean/quarantine/` | `/home/afr/reid-clean/quarantine/` |

每个来源只读选择扩展名为 `.log / .txt / .json / .pid` 且不超过 10 MiB 的普通文件，原目录层次、文件名和
mtime保持不变。checkpoint、tensor/cache、数据集、冻结pose、下载模型、虚拟环境和临时编译文件不属于Git日志
归档范围。

除上述权威来源外，还对`/home/afr`执行了全实验路径扫描，选择`.log`、`*log*.txt`、`test*.txt`与`.pid`，
排除Git、Conda、venv、site-packages、node_modules和runtime依赖目录。4,276条原始路径对应867份不同SHA；
权威目录未覆盖的521份内容以首个远端路径保存到`lab4090_unique/`。因此formal clone的重复日志通过路径清单
映射到同一内容，不在工作树中机械复制几十份。

## 清单

- `REMOTE_MANIFEST.tsv`：四列依次为归档来源键、字节数、远端SHA256、来源内相对路径。
- `REMOTE_ALL_TEXT_LOGS.tsv`：全远端扫描的字节数、SHA256和原始绝对路径，共4,276条。
- `REMOTE_UNIQUE_BLOBS.tsv`：权威目录未覆盖的521份去重内容及其Git归档路径。
- `REMOTE_UNIQUE_MISSING_PATHS.txt`：用于只读复制这些去重内容的远端相对路径。
- `MANIFEST.sha256`：本地归档文件的SHA256，可在本目录执行：

  ```bash
  shasum -a 256 -c MANIFEST.sha256
  ```

- `REMOTE_EXCLUDED_LARGE.tsv`：超过10 MiB、未嵌入Git的生成资产，记录字节数、SHA256和远端绝对路径。

本次唯一超过阈值的文件是exp392生成的62,938,306字节pose-map JSON。它不是训练/测试文本日志，已通过SHA索引，
未复制进Git。一个163字节的AppleDouble `._phase0a.runner.log`元数据文件因匹配远端日志路径被原样保留；
它是唯一含NUL字节的归档项，并由SHA清单覆盖。

## 完整性边界

该快照不修改、删除或续跑任何远端实验，也不包含checkpoint和数据资产。日志中的失败、用户终止、异常和不利点
均原样保留，没有筛除。
