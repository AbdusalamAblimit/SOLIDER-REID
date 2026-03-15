# exp064 Probabilistic Keypoint Embeddings (PKE) 训练监控

## 实验信息
- **方法**: PSG + GCN + PKE (per-keypoint Gaussian mu+sigma)
- **配置**: `configs/occluded_duke/pose_psg_gcn_pke.yml`
- **输出**: `log/occluded_duke/exp064_pke/`
- **对照**: exp030a (PSG+GCN) 3-seed = 60.73%/72.57%
- **核心改动**: GCN 输出 (mu, log_sigma)，test feature = concat(global, mu, log_sigma) = 2304-d
- **等待 review 通过后启动训练**
