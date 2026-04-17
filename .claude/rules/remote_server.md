# 远程服务器（3 台 5060 Ti 16G）

## SSH 别名

密钥已生成于 `~/.ssh/id_ed25519`，config 在 `~/.ssh/config`，公钥已推到 3 台：

```
srvA → i-2.gpushare.com:29162   # 旧机器，已部分配置
srvB → i-1.gpushare.com:61604   # 新机器
srvC → i-2.gpushare.com:25551   # 新机器
```

直接 `ssh srvA`、`ssh srvB`、`ssh srvC` 即可。**不要再用 sshpass**。

## 硬件 / 磁盘

| 别名 | GPU | /root | /hy-tmp | Python |
|------|-----|-------|---------|--------|
| srvA | 5060 Ti 16G | 30G (23G used) | 50G | system python3.11 + torch 2.9.1+cu128 |
| srvB | 5060 Ti 16G | 30G (fresh) | 50G | system python3.11 + torch 2.9.1+cu128 |
| srvC | 5060 Ti 16G | 30G (fresh) | 50G | system python3.11 + torch 2.9.1+cu128 |

**/root 吃紧（30G）**，不能把数据/日志放 /root。

## 远程目录约定

```
/root/work/SOLIDER-REID/         # git 仓库 + 代码
/root/work/SOLIDER-REID/data → /hy-tmp/data      # symlink
/root/work/SOLIDER-REID/log  → /hy-tmp/log       # symlink
/hy-tmp/data/occluded_duke/      # 本地数据（含 pose_data）
/hy-tmp/data/market1501/
/hy-tmp/data/occluded_reid/
/hy-tmp/data/occluded_posetrack_reid/
/hy-tmp/log/occluded_duke/exp{NNN}/  # 训练输出（.pth 不占 /root）
```

## Python 依赖

B/C 已装（pip, 系统 python）：
`torch==2.9.1+cu128 torchvision==0.24.1 numpy pillow opencv yacs timm tqdm scipy matplotlib pandas ftfy regex einops`

若需 `mmdet/mmpose/mmengine`（只有 `scripts/extract_pose.py` 用）在本地做，远程不需要。

## 启动远程训练

```bash
ssh srvX "cd /root/work/SOLIDER-REID && PYTHONUNBUFFERED=1 nohup python3 train.py \
  --config_file configs/occluded_duke/xxx.yml \
  OUTPUT_DIR /hy-tmp/log/occluded_duke/exp{NNN} \
  > /tmp/exp{NNN}.log 2>&1 &"
```

`Bash(run_in_background=true)` 本地启动会立刻收到完成通知。远程用 nohup 不阻塞 SSH 退出。

## 监控远程训练（替代 sleep）

用 `Monitor` 工具：

```bash
# 每条关键事件一个通知
ssh srvX "tail -F /tmp/exp{NNN}.log" | grep -E --line-buffered 'Epoch \[[0-9]+\]|mAP|R1|Traceback|OOM|Killed|Error'
```

注意：
- `grep --line-buffered` 必须加，否则管道缓冲会延后通知几分钟
- `-E` alternation 必须覆盖崩溃信号（Traceback/OOM/Killed），silence 不等于 success
- `tail -F` 大写 F 更健壮（follow by name）

## 数据同步

本地→远程：
```bash
rsync -az --partial --info=progress2 \
  /root/work/data/Occluded_Duke/ srvX:/hy-tmp/data/occluded_duke/
```

远程→本地：
```bash
rsync -az --partial srvX:/hy-tmp/log/ /root/work/SOLIDER-REID/log_remote_srvX_backup/
```

Local→remote 带宽约 2 MB/s。大传输必须 `Bash(run_in_background=true)` 或 `nohup`。

## OSS（恒源云）

- 账号 `17602295205 / Xa401641` 目前 login 失败（"账号未注册"）
- `oss login -cloud=ent` cert 过期
- 暂时不用 OSS，全靠 rsync
- 若用户之后提供可用 OSS key，在这里更新

## 多机分工

本地 3090 + srvA/B/C 三台 5060 Ti = 4 个独立训练 slot。**任一 GPU 空闲立即开下一个实验**。

- 不同机器同时训不同实验（不同 PSG stage 配置 / 不同 GCN 容量 / 不同数据集）
- **绝不把同一实验的不同 seed 放不同机器凑 multi-seed 主线**，除非用户明确让做
- 机器空下来但文档没补：补文档不是闲着

## 新机器（未来）配置速查

```bash
# 1. 推公钥
sshpass -p '<passwd>' ssh -p <port> -o StrictHostKeyChecking=no -o PubkeyAuthentication=no root@<host> \
  "mkdir -p ~/.ssh && echo '$(cat ~/.ssh/id_ed25519.pub)' >> ~/.ssh/authorized_keys"

# 2. ~/.ssh/config 加别名 srvX

# 3. 装 rsync
ssh srvX "apt-get install -y rsync"

# 4. 装 pip deps
ssh srvX "pip3 install yacs timm tqdm scipy matplotlib pandas ftfy regex einops"

# 5. 克隆仓库
ssh srvX "cd /root/work && git clone --depth=1 -b exp/pose_heatmap https://github.com/AbdusalamAblimit/SOLIDER-REID.git"

# 6. 目录软链
ssh srvX "mkdir -p /hy-tmp/data /hy-tmp/log && cd /root/work/SOLIDER-REID && rm -rf data log && ln -s /hy-tmp/data data && ln -s /hy-tmp/log log"

# 7. rsync pretrained + datasets（后台）
nohup rsync -az --partial /root/work/SOLIDER-REID/pretrained/swin_*.pth /root/work/SOLIDER-REID/pretrained/clip_part_text_features.pt srvX:/root/work/SOLIDER-REID/pretrained/ &
nohup rsync -az --partial /root/work/data/Occluded_Duke/ srvX:/hy-tmp/data/occluded_duke/ &
```
