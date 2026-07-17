#!/bin/bash
# LM-S4 (lattice factor: phase/bbox/zoom isolated) + LM-S2-strong (richer TTA defense).
# Runs sequentially on the no-LM-loss ckpt, reusing the cached HR gallery.
cd /home/afr/SOLIDER-REID
PY=/usr/local/anaconda3/envs/mmpose-abu/bin/python
CK=log/market1501/exp359_abl_noLMloss/transformer_40.pth
KS=experiments/cargo_cvpb/cvpb_lattice_killswitch.py
G=/tmp/lattice_gallery_lmS2.npz
for AX in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 $PY $KS --ckpt $CK --heights 16 --K 9 --lattice_axis $AX --reuse_gallery --cache_gallery $G > /tmp/lmS4_ax$AX.log 2>&1
done
CUDA_VISIBLE_DEVICES=0 $PY $KS --ckpt $CK --heights 12 16 24 --K 9 --strong_tta --reuse_gallery --cache_gallery $G > /tmp/lmS2strong.log 2>&1
echo DONE > /tmp/lmS3S4_done.flag
