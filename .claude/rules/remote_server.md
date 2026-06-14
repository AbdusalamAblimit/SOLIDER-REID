# 远程服务器（3 台机器 / 4 个 GPU slot）

本地 Mac 只做编排和开发，**没有训练 GPU**。所有训练都在下面三台远程机器上跑。

## SSH 别名

密钥与 config 都在 `~/.ssh/config`，直接 `ssh <别名>`，**不要用 sshpass / 密码**。

| 别名 | 全名（config 里的 Host） | 连接方式 | 状态（2026-06-14 探测） |
|------|--------------------------|----------|--------------------------|
| `lab-3090-d` | `lab-3090-d` | ProxyJump `lab-3090`(Tailscale 100.95.201.91, user server) → 172.17.0.2 docker, root | ✅ 可连（2026-06-14 一次连上，container 18fbbab202e1）；跳板偶尔 banner 超时，重试即可 |
| `lab-4090` | `lab-4090` / `lab4090` | ProxyJump `relay4090`(Tailscale 100.94.229.1, user administrator) → 10.0.70.128, root | ✅ 可连（2026-06-14 重试 2 次连上）；**第一次 banner 超时正常，多试几次会恢复**，别判定下线 |
| `hyy` | `hyy-5060ti-double` | 直连 i-1.gpushare.com:17660, root | ✅ 在线，已验证 |

> 短写 `hyy` 实际别名是 `hyy-5060ti-double`，`4090` 实际是 `lab-4090`/`lab4090`。下文 `<srv>` 指这三者之一。

## 硬件 / GPU slot

| 机器 | GPU | slot 数 | Python / torch | 备注 |
|------|-----|---------|----------------|------|
| `lab-3090-d` | RTX 3090 24G | 1 | py3.10.12 + torch 2.7.1+cu118 | lab 3090 上的 docker 容器（home /root）；`/` 1.8T，716G free |
| `lab-4090` | RTX 4090 D 24G | 1 | py3.10.12（torch 在 conda/venv，非系统 python） | **共享多用户机**（/home 下 afr/hbj/cjy），GPU 可能被别人占，开实验前先 `nvidia-smi` 看占用；登录 user=root，实验在 `/home/afr/` 下 |
| `hyy` | **RTX 5060 Ti 16G × 2** | 2 | py3.11.12 + torch 2.9.1+cu128 | 双卡，可同时跑 2 个实验（GPU 0 / GPU 1） |

合计 **4 个独立训练 slot**（3090 ×1 + 4090 ×1 + 5060 Ti ×2）。

> `lab-3090-d` / `lab-4090` 探测时不可达，硬件细节（磁盘、python、torch、目录）**首次连通后在此表补全**，不要凭空填写。

## 磁盘 / 目录约定

### hyy（已验证）

```
/hy-tmp/reid-clean/SOLIDER-REID/        # git 仓库 + 代码（branch exp/pose_heatmap）
/hy-tmp/reid-clean/data/Occluded_Duke/  # 数据
/hy-tmp/reid-clean/data/market1501/
/hy-tmp/reid-clean/SOLIDER-REID/log/    # 仓库内 log 目录（训练输出）
/hy-tmp/*.log                           # nohup 日志（exp{NNN}_*.log / hyy_*.out）
```

磁盘：`/`(overlay) 30G（8.5G used，吃紧，别往 /root 放数据/ckpt）；`/hy-tmp` 50G（2026-06-14 清理 log/checkpoints 后已用 ~23G）。注意 `.venv`(7.4G)+`.uvcache`(7G) 占 ~14G，需要更多空间时可 `uv cache clean`。

### lab-4090（已确认，2026-06-14）

**实验目录 = `/home/afr/SOLIDER-REID`**（用户确认：afr 里直接那个，**不是 reid-clean**）。branch `exp/pose_heatmap @ 131e8f6`，与本地仓库同一条线，occluded_duke / market 主线。

```
/home/afr/SOLIDER-REID/            # ★ 实验目录（继续做实验就用这个）
/home/afr/SOLIDER-REID/log/        # 训练输出（85G，occluded_duke / market1501 / od_tiny_*_clean）
/home/afr/SOLIDER-REID/data/       # 自带数据集 market/MSMT17/occluded_duke/occluded_reid/VOCdevkit
/mnt1/afrdata/                     # 共享数据集挂载（备用）
/home/afr/reid-clean/             # ✗ 另一条线（lifelong / 换衣 ReID, AAAI2024-LSTKC），不是我们的实验目录，别动也别删
```

- `/` 1.8T（338G free，81% used）。**登录是 root 但代码归 afr（uid 501）**，root 读写没问题，但注意别破坏 afr 的环境。
- 共享机：`/home/{afr,hbj,cjy}` 多人 + GPU 共用，开训前 `nvidia-smi` 确认 4090 空闲。

### lab-3090-d（已确认，2026-06-14）

**实验目录 = `/root/work/SOLIDER-REID`**（branch `exp/pose_heatmap`，remote = AbdusalamAblimit/SOLIDER-REID，与本地同一条线）。同 lab-4090：用主 checkout，**不是 reid-clean**。

```
/root/work/SOLIDER-REID/            # ★ 实验目录（继续做实验就用这个）
/root/work/SOLIDER-REID/log/        # 训练输出（99G，occluded_duke / market1501 / occluded_posetrack）
/root/work/SOLIDER-REID/data/       # 自带数据集 occluded_duke/market1501/occluded_posetrack_reid/occluded_reid/MSMT17/VOCdevkit
/root/work/SOLIDER-REID/solider-logs/  # SOLIDER backbone 旧日志（txt，3 月）
/root/reid-clean/SOLIDER-REID/      # ✗ 另一条线（reid-clean，posetrack/grp_pose），不是实验目录，别动也别删
/root/work/SOLIDER-REID-origin/     # ✗ 上游 origin 参考(master@8c08e1c, 2023)，勿用
```

- ⚠️ `/root/work/SOLIDER-REID/.git/config` 的 remote URL **明文存了 GitHub PAT**，暴露在容器里。提醒用户吊销/轮换，改用 SSH key 或 credential helper。
- `/` 1.8T，716G free，磁盘宽裕。

## 启动远程训练

```bash
ssh <srv> "cd <repo> && PYTHONUNBUFFERED=1 nohup python3 train.py \
  --config_file configs/occluded_duke/xxx.yml \
  OUTPUT_DIR <repo>/log/occluded_duke/exp{NNN} \
  > /tmp/exp{NNN}.log 2>&1 &"
```

hyy 双卡指定 GPU：在命令前加 `CUDA_VISIBLE_DEVICES=0`（或 `=1`），两条独立实验各占一卡。

远程用 `nohup` 不阻塞 SSH 退出。本地 `Bash(run_in_background=true)` 启动会立刻收到完成通知。

## 监控远程训练（替代 sleep）

用 `Monitor` 工具，关键事件即时推送：

```bash
ssh <srv> "tail -F /tmp/exp{NNN}.log" \
  | grep -E --line-buffered 'Epoch \[[0-9]+\]|mAP|R1|Traceback|OOM|Killed|Error'
```

注意：
- `grep --line-buffered` 必须加，否则管道缓冲会延后通知几分钟
- `-E` alternation 必须覆盖崩溃信号（Traceback/OOM/Killed），silence ≠ success
- `tail -F` 大写 F 更健壮（follow by name）
- hyy `/` 只有 30G，eval 阶段内存峰值高（历史上 5060 Ti 16G 有 e100 eval OOM-killed），盯紧 OOM/Killed

## 数据同步

机器间 / 推数据用 rsync over SSH（断点续传 `--partial`）：

```bash
# 推数据到某机
rsync -az --partial --info=progress2 <源路径>/ <srv>:<目标数据目录>/
# 拉日志回本地备份
rsync -az --partial <srv>:<repo>/log/ ./log_remote_<srv>_backup/
```

大传输（带宽约 2 MB/s）必须 `Bash(run_in_background=true)` 或远程 `nohup`，配 Monitor 跟踪。

## OSS（恒源云，仅 hyy 这类 gpushare 机器相关）

- 账号登录历史上失败（"账号未注册" / ent cert 过期），**暂时不用 OSS，全靠 rsync**。
- 若用户提供可用 OSS 凭证，在此更新。
- lab-3090-d / lab-4090 是实验室机器，与 OSS 无关。

## 多机分工

4 个 slot（lab-3090-d ×1、lab-4090 ×1、hyy ×2）。**任一 GPU 空闲立即开下一个实验**。

- 不同机器 / hyy 两张卡同时训不同实验（不同 PSG stage / 不同 GCN 容量 / 不同数据集）
- **绝不把同一实验的不同 seed 拆到不同机器凑 multi-seed 主线**，除非用户明确要求
- 机器空下来但文档没补：补文档不是闲着
- 跨设备方差：历史上 5060 Ti vs 3090/4090 结果 Δ<0.5%，可互信，但论文主表数字标清在哪台跑的
