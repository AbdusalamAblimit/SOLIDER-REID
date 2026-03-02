# SOLIDER-REID Environment Setup Guide

## System Requirements

| Item | Requirement |
|------|-------------|
| OS | Ubuntu 22.04 LTS (x86_64) |
| GPU | NVIDIA GPU (tested on RTX 3090) |
| NVIDIA Driver | >= 525.85.12 (CUDA Driver 12.0) |
| Disk | >= 15 GB free space |

## 1. Install Miniconda

Download and install Miniconda (silent mode):

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
rm /tmp/miniconda.sh
```

Initialize conda for your shell:

```bash
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc
```

Accept Terms of Service (Miniconda 25.x+ requires this):

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

Verify:

```bash
conda --version
# conda 25.11.1
```

## 2. Create Conda Environment

Create a new environment with Python 3.8:

```bash
conda create -n solider-reid python=3.8 -y
conda activate solider-reid
```

> **Note:** Python 3.8 is chosen for compatibility with PyTorch 1.13.1 and OpenMMLab 2.x packages.

## 3. Install PyTorch (CUDA 11.7)

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117
```

Verify:

```bash
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())"
# 1.13.1+cu117 | CUDA: True
```

> **Note:** Although the system NVIDIA driver supports CUDA 12.0, PyTorch's bundled CUDA 11.7 runtime works fine because the driver is backward-compatible.

## 4. Install OpenMMLab (MMPose) Ecosystem

MMPose and its dependencies are installed using `openmim`, the official OpenMMLab package manager. The installation order matters — install from bottom (mmengine) to top (mmpose/mmpretrain).

### 4.1 Install openmim

```bash
pip install -U openmim
```

### 4.2 Install MM packages (in order)

```bash
mim install "mmengine==0.10.7"
mim install "mmcv==2.1.0"
mim install "mmdet==3.2.0"
mim install "mmpose==1.3.2"
mim install "mmpretrain==1.2.0"
```

Verify:

```bash
python -c "
import mmengine; print('mmengine:', mmengine.__version__)
import mmcv;     print('mmcv:',     mmcv.__version__)
import mmdet;    print('mmdet:',    mmdet.__version__)
import mmpose;   print('mmpose:',   mmpose.__version__)
import mmpretrain; print('mmpretrain:', mmpretrain.__version__)
"
```

Expected output:

```
mmengine: 0.10.7
mmcv: 2.1.0
mmdet: 3.2.0
mmpose: 1.3.2
mmpretrain: 1.2.0
```

### Dependency Graph

```
mmpose (1.3.2) ─────┐
mmpretrain (1.2.0) ──┤
mmdet (3.2.0) ───────┤
                     ├── mmcv (2.1.0) ── mmengine (0.10.7)
                     │
                     └── PyTorch (1.13.1+cu117)
```

## 5. Install Additional Dependencies

```bash
pip install tensorboard yacs IPython timm
```

## 6. Verify Complete Environment

Run the following script to verify all modules and the project itself load correctly:

```bash
cd /path/to/SOLIDER-REID
python -c "
import torch
print('=== PyTorch ===')
print('  Version:', torch.__version__)
print('  CUDA:', torch.cuda.is_available(), '| ver:', torch.version.cuda)
if torch.cuda.is_available():
    print('  GPU:', torch.cuda.get_device_name(0))

import mmengine, mmcv, mmdet, mmpose, mmpretrain
print('=== OpenMMLab ===')
print('  mmengine:', mmengine.__version__)
print('  mmcv:', mmcv.__version__)
print('  mmdet:', mmdet.__version__)
print('  mmpose:', mmpose.__version__)
print('  mmpretrain:', mmpretrain.__version__)

import tensorboard, yacs, IPython, timm
print('=== Others ===')
print('  tensorboard:', tensorboard.__version__)
print('  IPython:', IPython.__version__)
print('  timm:', timm.__version__)

print()
print('=== Project Modules ===')
from config import cfg;           print('  config: OK')
from model import make_model;     print('  model: OK')
from utils.logger import setup_logger; print('  utils: OK')
print()
print('All imports successful!')
"
```

Expected output:

```
=== PyTorch ===
  Version: 1.13.1+cu117
  CUDA: True | ver: 11.7
  GPU: NVIDIA GeForce RTX 3090
=== OpenMMLab ===
  mmengine: 0.10.7
  mmcv: 2.1.0
  mmdet: 3.2.0
  mmpose: 1.3.2
  mmpretrain: 1.2.0
=== Others ===
  tensorboard: 2.14.0
  IPython: 8.12.3
  timm: 1.0.25

=== Project Modules ===
  config: OK
  model: OK
  utils: OK

All imports successful!
```

## Key Package Versions Summary

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.8.20 | Runtime |
| PyTorch | 1.13.1+cu117 | Deep learning framework |
| torchvision | 0.14.1+cu117 | Image transforms & datasets |
| mmengine | 0.10.7 | OpenMMLab base engine |
| mmcv | 2.1.0 | OpenMMLab CV utilities |
| mmdet | 3.2.0 | Object detection (mmpose dep) |
| mmpose | 1.3.2 | Pose estimation backbone |
| mmpretrain | 1.2.0 | Pre-trained model zoo |
| tensorboard | 2.14.0 | Training visualization |
| yacs | 0.1.8 | Config system |
| timm | 1.0.25 | Transformer model library |
| IPython | 8.12.3 | Interactive debugging |

## One-Click Setup Script

For convenience, here is the entire process as a single script:

```bash
#!/bin/bash
set -e

# 1. Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
rm /tmp/miniconda.sh
export PATH="$HOME/miniconda3/bin:$PATH"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 2. Create environment
conda create -n solider-reid python=3.8 -y
conda activate solider-reid

# 3. Install PyTorch
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# 4. Install OpenMMLab ecosystem
pip install -U openmim
mim install "mmengine==0.10.7"
mim install "mmcv==2.1.0"
mim install "mmdet==3.2.0"
mim install "mmpose==1.3.2"
mim install "mmpretrain==1.2.0"

# 5. Install additional dependencies
pip install tensorboard yacs IPython timm

echo "Environment setup complete!"
```

## Troubleshooting

### Q: `conda tos accept` command not recognized
Older versions of conda don't require TOS acceptance. This step can be skipped if your conda version < 25.x.

### Q: Network timeout during PyTorch download
PyTorch wheel is ~1.8GB. Use `--default-timeout=300` flag or retry:
```bash
pip install --default-timeout=300 torch==1.13.1+cu117 ...
```

### Q: `mim install mmcv` is very slow or compiling from source
Ensure PyTorch and CUDA versions match. `mim` will try to find a pre-built wheel; if versions don't match, it falls back to building from source (can take 30+ minutes).

### Q: CUDA out of memory during training
Use gradient checkpointing by adding `MODEL.WITH_CP True` in the config or command line arguments.
