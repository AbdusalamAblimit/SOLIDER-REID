#!/usr/bin/env python3
"""Diagnose PoseBackboneModel forward output structure (where is featmaps for stripe-pool)."""
import sys, os, numpy as np
sys.argv = ['ks', '--ckpt', 'log/market1501/exp359_abl_noLMloss/transformer_40.pth',
            '--config', 'configs/market/pose_psg_lgpa_gcn_base.yml', '--dataset', 'market1501']
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvpb_lattice_killswitch as ks
from datasets.bases import read_image
import torch

ext = ks.FrozenExtractor()
q = ks.list_split(os.path.join(ks._repo, 'data', 'market1501', 'query'))[:1]
im = ks._to_target_aspect(read_image(q[0][0]))
arr = np.stack([ks.pil_to_tensor_np(im)], 0)
t = torch.from_numpy(arr).cuda()
cam = torch.zeros(1, dtype=torch.long, device=t.device)
view = torch.zeros(1, dtype=torch.long, device=t.device)
with torch.no_grad():
    out = ext.model(t, cam_label=cam, view_label=view, pose_dict=None)
print('out type:', type(out))
if isinstance(out, (tuple, list)):
    print('out len:', len(out))
    for i, o in enumerate(out):
        if torch.is_tensor(o):
            print(f'  out[{i}]: tensor {tuple(o.shape)}')
        else:
            print(f'  out[{i}]: {type(o)} = {o if not hasattr(o, "shape") else o.shape}')
else:
    print('single:', getattr(out, 'shape', out))
print('[diag done]')
