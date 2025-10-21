#!/usr/bin/env python3
import re, sys, os, json
from glob import glob

LOG_RE = re.compile(r'(Rank-1|mAP)\s*[:=]\s*([0-9.]+)')
CFG_KEYS = [
    ("MODEL.POSE.FUSION_MODE", r"MODEL\.POSE\.FUSION_MODE\s+(\w+)"),
    ("MODEL.POSE.FUSE_STAGE", r"MODEL\.POSE\.FUSE_STAGE\s+(\d+)"),
    ("MODEL.POSE.SCALE",      r"MODEL\.POSE\.SCALE\s+([0-9.]+)"),
    ("MODEL.POSE.HM_NORM",    r"MODEL\.POSE\.HM_NORM\s+(\w+)"),
    ("MODEL.POSE.USE_VIS",    r"MODEL\.POSE\.USE_VIS\s+(true|false)"),
    ("MODEL.POSE.DETACH",     r"MODEL\.POSE\.DETACH\s+(true|false)"),
]

def read_tail(path, n=4000):
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            off = max(0, size - n*80)
            f.seek(off)
            data = f.read().decode('utf-8', errors='ignore')
        return data
    except Exception:
        return ""

def parse_one(log_path):
    txt = read_tail(log_path, n=8000)
    # metrics: 取最后一次出现
    metrics = {}
    for m in LOG_RE.finditer(txt):
        metrics[m.group(1)] = float(m.group(2))
    # cfg (从命令行 echo 的覆盖项里抓)
    cfg = {}
    for key, pat in CFG_KEYS:
        m = re.search(pat, txt)
        if m:
            cfg[key] = m.group(1)
    return metrics, cfg

def main(root):
    rows = []
    for out_dir in sorted(glob(os.path.join(root, "*"))):
        log = os.path.join(out_dir, "train.log")
        if not os.path.isfile(log):
            continue
        metrics, cfg = parse_one(log)
        row = {
            "exp_dir": out_dir,
            "Rank-1": metrics.get("Rank-1", -1),
            "mAP": metrics.get("mAP", -1),
        }
        row.update(cfg)
        rows.append(row)

    # 排序：先按 mAP，再按 Rank-1
    rows.sort(key=lambda r: (-r["mAP"], -r["Rank-1"]))
    # 输出 CSV
    hdr = ["mAP", "Rank-1", "MODEL.POSE.FUSION_MODE", "MODEL.POSE.FUSE_STAGE",
           "MODEL.POSE.SCALE", "MODEL.POSE.HM_NORM", "MODEL.POSE.USE_VIS",
           "MODEL.POSE.DETACH", "exp_dir"]
    print(",".join(hdr))
    for r in rows:
        print(",".join(str(r.get(k, "")) for k in hdr))

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "./log/msmt17/msmt_pose_sweep"
    main(root)
