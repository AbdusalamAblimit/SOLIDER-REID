#!/usr/bin/env python3
"""Summarize all SPTrans experiment results under a log directory.

Usage:
    python summarize_experiments.py [LOG_DIR]

Default LOG_DIR: ./log/sptrans_v2

Output: a single text block you can copy-paste to Claude for analysis.
"""

import os
import re
import sys
from collections import defaultdict

LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else "./log/sptrans_v2"


def parse_train_log(path):
    """Parse train_log.txt and extract per-epoch metrics."""
    epochs = {}  # epoch -> {branch -> {metric -> value}}
    loss_curve = []  # [(epoch, iter, loss, acc, lr)]
    current_epoch = None
    current_branch = None

    with open(path, "r") as f:
        for line in f:
            # Loss line: Epoch[30] Iter[100/186] Loss: 3.456, Acc: 0.567, Base Lr: 1.23e-04
            m = re.search(
                r"Epoch\[(\d+)\] Iter\[(\d+)/(\d+)\] Loss: ([\d.]+), Acc: ([\d.]+), Base Lr: ([\d.e+-]+)",
                line,
            )
            if m:
                ep = int(m.group(1))
                it = int(m.group(2))
                total = int(m.group(3))
                loss = float(m.group(4))
                acc = float(m.group(5))
                lr = m.group(6)
                # Only record last iter of each epoch for the curve
                if it >= total - 20:
                    loss_curve.append((ep, loss, acc, lr))
                continue

            # Validation Results [global] - Epoch: 120
            m = re.search(r"Validation Results \[(\w+)\] - Epoch: (\d+)", line)
            if m:
                current_branch = m.group(1)
                current_epoch = int(m.group(2))
                if current_epoch not in epochs:
                    epochs[current_epoch] = {}
                if current_branch not in epochs[current_epoch]:
                    epochs[current_epoch][current_branch] = {}
                continue

            # mAP: 91.5%
            m = re.search(r"mAP: ([\d.]+)%", line)
            if m and current_epoch is not None and current_branch is not None:
                epochs[current_epoch][current_branch]["mAP"] = float(m.group(1))
                continue

            # CMC curve [global], Rank-1  :96.3%
            m = re.search(r"Rank-(\d+)\s*:\s*([\d.]+)%", line)
            if m and current_epoch is not None and current_branch is not None:
                rank = int(m.group(1))
                val = float(m.group(2))
                epochs[current_epoch][current_branch][f"R{rank}"] = val
                continue

    return epochs, loss_curve


def summarize():
    if not os.path.isdir(LOG_DIR):
        print(f"ERROR: {LOG_DIR} not found")
        sys.exit(1)

    experiments = sorted(os.listdir(LOG_DIR))
    if not experiments:
        print(f"No experiments found in {LOG_DIR}")
        sys.exit(1)

    print("=" * 80)
    print(f"EXPERIMENT SUMMARY — {os.path.abspath(LOG_DIR)}")
    print("=" * 80)

    final_table = []

    for exp_name in experiments:
        exp_dir = os.path.join(LOG_DIR, exp_name)
        log_path = os.path.join(exp_dir, "train_log.txt")
        if not os.path.isfile(log_path):
            continue

        epochs, loss_curve = parse_train_log(log_path)

        print(f"\n{'─' * 70}")
        print(f"  {exp_name}")
        print(f"{'─' * 70}")

        # Status
        if not epochs:
            max_ep = 0
            if loss_curve:
                max_ep = loss_curve[-1][0]
            print(f"  Status: TRAINING (Epoch {max_ep}, no eval yet)")
            continue

        max_epoch = max(epochs.keys())
        is_done = max_epoch >= 120
        status = "DONE" if is_done else f"IN PROGRESS (Epoch {max_epoch})"
        print(f"  Status: {status}")

        # Loss curve (sampled)
        if loss_curve:
            print(f"\n  Loss curve (end-of-epoch):")
            print(f"  {'Epoch':>6}  {'Loss':>8}  {'Acc':>8}  {'LR':>12}")
            # Sample: first, every 10th, last
            seen = set()
            samples = []
            for ep, loss, acc, lr in loss_curve:
                if ep not in seen and (ep <= 1 or ep % 10 == 0 or ep == loss_curve[-1][0]):
                    samples.append((ep, loss, acc, lr))
                    seen.add(ep)
            for ep, loss, acc, lr in samples:
                print(f"  {ep:>6}  {loss:>8.3f}  {acc:>7.1%}  {lr:>12}")

        # Eval results at key epochs
        eval_epochs = sorted(epochs.keys())
        if eval_epochs:
            branches = set()
            for ep_data in epochs.values():
                branches.update(ep_data.keys())
            branches = sorted(branches)

            print(f"\n  Eval results:")
            header = f"  {'Epoch':>6}"
            for b in branches:
                header += f"  {b+' mAP':>12}  {b+' R1':>10}"
            print(header)

            for ep in eval_epochs:
                row = f"  {ep:>6}"
                for b in branches:
                    if b in epochs[ep]:
                        mAP = epochs[ep][b].get("mAP", 0)
                        r1 = epochs[ep][b].get("R1", 0)
                        row += f"  {mAP:>11.1f}%  {r1:>9.1f}%"
                    else:
                        row += f"  {'—':>12}  {'—':>10}"
                print(row)

        # Collect final results for summary table
        if is_done and 120 in epochs:
            entry = {"name": exp_name}
            for b in epochs[120]:
                entry[f"{b}_mAP"] = epochs[120][b].get("mAP", 0)
                entry[f"{b}_R1"] = epochs[120][b].get("R1", 0)
            final_table.append(entry)

    # Final summary table
    if final_table:
        print(f"\n{'=' * 80}")
        print("FINAL RESULTS (Epoch 120)")
        print(f"{'=' * 80}")

        # Collect all branch keys
        all_keys = set()
        for e in final_table:
            all_keys.update(k for k in e.keys() if k != "name")
        branch_names = sorted(set(k.split("_")[0] for k in all_keys if "_mAP" in k))

        header = f"{'Experiment':<40}"
        for b in branch_names:
            header += f"  {b+' mAP':>10}  {b+' R1':>8}"
        print(header)
        print("─" * len(header))

        for e in final_table:
            row = f"{e['name']:<40}"
            for b in branch_names:
                mAP = e.get(f"{b}_mAP", 0)
                r1 = e.get(f"{b}_R1", 0)
                if mAP > 0:
                    row += f"  {mAP:>9.1f}%  {r1:>7.1f}%"
                else:
                    row += f"  {'—':>10}  {'—':>8}"
            print(row)

    print(f"\n{'=' * 80}")
    print("END OF SUMMARY")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    summarize()
