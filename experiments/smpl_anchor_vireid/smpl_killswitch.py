# SMPL-anchor kill-switch: 2D pose 在 RegDB IR(热)图上能不能提?
# 用 torchvision keypointrcnn(COCO 2D 人体关键点)对比 RGB-visible vs IR-thermal。
# 若 IR 检测率/关键点置信远低于 RGB → 热图上提不出人体几何 → SMPL 锚高风险 → 转 Swin-VI 机制。
import torch, torchvision, glob
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

device = 'cuda'
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights='DEFAULT', weights_backbone=None).to(device).eval()

def eval_set(paths, name):
    dets05, dets07, confs = 0, 0, []
    for p in paths:
        img = Image.open(p).convert('RGB')
        x = TF.to_tensor(img).to(device)
        with torch.no_grad():
            out = model([x])[0]
        if len(out['scores']):
            s = out['scores'][0].item()
            if s > 0.5: dets05 += 1
            if s > 0.7: dets07 += 1
            if s > 0.5:
                confs.append(out['keypoints_scores'][0].mean().item())
    n = len(paths)
    print(f"{name}: n={n} | det@0.5={dets05}({100*dets05/n:.0f}%) det@0.7={dets07}({100*dets07/n:.0f}%) | kp_conf(det>0.5)={np.mean(confs) if confs else 0:.2f}")
    return dets05/n, np.mean(confs) if confs else 0

RGB = sorted(glob.glob('/home/afr/vireid/data/RegDB/Visible/*/*.bmp'))
IR  = sorted(glob.glob('/home/afr/vireid/data/RegDB/Thermal/*/*.bmp'))
print(f"found RGB={len(RGB)} IR={len(IR)}")
RGB_s = RGB[::max(1, len(RGB)//150)][:150]
IR_s  = IR[::max(1, len(IR)//150)][:150]
r_det, r_conf = eval_set(RGB_s, 'RGB-visible')
i_det, i_conf = eval_set(IR_s,  'IR-thermal ')
print("="*50)
print(f"判定: IR/RGB 检测率比 = {i_det/max(r_det,1e-6):.2f}, 置信比 = {i_conf/max(r_conf,1e-6):.2f}")
print("IR 检测率 >70% 且置信比 >0.7 → pose-on-IR 可用 → SMPL 锚有戏(过)")
print("IR 检测率 <50% 或置信比 <0.5 → 热图提不出几何 → SMPL 锚高风险(死, 转 Swin-VI)")
