---
tags:
- image-classification
- ecology
- animals
- re-identification
library_name: wildlife-datasets
license: cc-by-nc-4.0
---
# Model card for MegaDescriptor-L-384

A Swin-L image feature model. Superwisely pre-trained on animal re-identification datasets.


## Model Details
- **Model Type:** Animal re-identification / feature backbone
- **Model Stats:**
  - Params (M): 228.8
  - Image size: 384 x 384
  - Architecture: swin_large_patch4_window12_384
- **Paper:** [WildlifeDatasets_An_Open-Source_Toolkit_for_Animal_Re-Identification](https://openaccess.thecvf.com/content/WACV2024/html/Cermak_WildlifeDatasets_An_Open-Source_Toolkit_for_Animal_Re-Identification_WACV_2024_paper.html)
- **Related Papers:**
  - [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
  - [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/pdf/2304.07193.pdf)
- **Pretrain Dataset:** All available re-identification datasets --> https://github.com/WildlifeDatasets/wildlife-datasets

## Model Usage
### Image Embeddings
```python

import timm
import torch
import torchvision.transforms as T

from PIL import Image
from urllib.request import urlopen

model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
model = model.eval()

train_transforms = T.Compose([T.Resize(size=(384, 384)),
                              T.ToTensor(), 
                              T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) 

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

output = model(train_transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor
# output is a (1, num_features) shaped tensor
```

## Citation

```bibtex
@inproceedings{vcermak2024wildlifedatasets,
  title={WildlifeDatasets: An open-source toolkit for animal re-identification},
  author={{\v{C}}erm{\'a}k, Vojt{\v{e}}ch and Picek, Lukas and Adam, Luk{\'a}{\v{s}} and Papafitsoros, Kostas},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5953--5963},
  year={2024}
}
```