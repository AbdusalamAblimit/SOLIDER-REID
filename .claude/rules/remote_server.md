# 远程服务器（恒源云 5060 Ti）

## 连接方式

```bash
sshpass -p 'aZKBF3qdSS59Wf4uveVQgEwWAtHAwbeg' ssh -p 29162 -o StrictHostKeyChecking=no root@i-2.gpushare.com
```

## 启动远程训练

```bash
sshpass -p 'aZKBF3qdSS59Wf4uveVQgEwWAtHAwbeg' ssh -p 29162 -o StrictHostKeyChecking=no root@i-2.gpushare.com \
  "echo '#!/bin/bash
cd /root/work/SOLIDER-REID
PYTHONUNBUFFERED=1 python3 train.py --config_file {CONFIG} OUTPUT_DIR {OUTPUT}' > /tmp/run.sh && \
chmod +x /tmp/run.sh && nohup /tmp/run.sh > /tmp/train_remote.log 2>&1 &"
```

## 同步代码到远程

```bash
git push origin exp/pose_heatmap
sshpass -p 'aZKBF3qdSS59Wf4uveVQgEwWAtHAwbeg' ssh -p 29162 -o StrictHostKeyChecking=no root@i-2.gpushare.com \
  'git -C /root/work/SOLIDER-REID pull origin exp/pose_heatmap'
```

## 注意事项

- GPU: NVIDIA 5060 Ti
- 项目路径: `/root/work/SOLIDER-REID`
- 数据路径: `data/occluded_duke`（已就位）
- SSH 默认在 `/root`，必须 cd 到项目目录才能运行
- **本地和远程不能跑几乎一样的实验**，必须是不同创新点或强对照
