#!/usr/bin/env python3
"""exp323 — MLLM-as-occlusion-reasoner A/B validation (codex / GPT-5.5 multimodal).

Tests whether injecting pose-visibility grounding (prompt B) improves the strong
MLLM's same/different identity judgment on heavily-occluded ReID pairs vs a naked
prompt (A). Red-line #6 defense: the gain must concentrate on heavier occlusion.

Modes:
  select  : pick a balanced subsample stratified by occlusion level -> sub_pairs.json + image filelists
  run     : drive codex A/B per pair (parallel, resumable) -> results.jsonl
  analyze : accuracy A vs B overall + by n_visible bucket + McNemar-style flip counts

Images expected at experiments/exp323/_imgs/{query,gallery}/<fname> (rsync from box between select and run).
"""
import os, sys, json, argparse, subprocess, random
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = os.path.join(HERE, "pairs_heavyocc.json")
SUB = os.path.join(HERE, "sub_pairs.json")
RESULTS = os.path.join(HERE, "results.jsonl")
IMG_Q = os.path.join(HERE, "_imgs", "query")
IMG_G = os.path.join(HERE, "_imgs", "gallery")
CODEX = "/opt/homebrew/bin/codex"
PROXY = "http://127.0.0.1:7890"
PARTS = ["head", "torso", "arms", "legs", "feet"]


def load_pairs():
    d = json.load(open(PAIRS))
    if isinstance(d, dict):
        # dict has metadata + a list of pairs somewhere; find the list
        for v in d.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "gt_same" in v[0]:
                return v
        raise SystemExit("no pair list found in dict")
    return d


def select(n_per_side=150, max_per_query=2, seed=42):
    pairs = load_pairs()
    rng = random.Random(seed)
    rng.shuffle(pairs)
    # stratify by n_visible bucket (0..8), balance same/diff, cap per query
    buckets = {}
    for p in pairs:
        buckets.setdefault(p["query_n_visible"], {"same": [], "diff": []})
        buckets[p["query_n_visible"]]["same" if p["gt_same"] else "diff"].append(p)
    chosen, qcount = [], {}
    bvals = sorted(buckets)
    per_bucket = max(1, n_per_side // max(1, len(bvals)))
    for side in ("same", "diff"):
        for b in bvals:
            took = 0
            for p in buckets[b][side]:
                q = p["query_img"]
                if qcount.get(q, 0) >= max_per_query:
                    continue
                chosen.append(p); qcount[q] = qcount.get(q, 0) + 1; took += 1
                if took >= per_bucket:
                    break
    # top up to balance if short
    random.Random(seed + 1).shuffle(chosen)
    json.dump(chosen, open(SUB, "w"), ensure_ascii=False, indent=0)
    qs = sorted({p["query_img"] for p in chosen})
    gs = sorted({p["gallery_img"] for p in chosen})
    open(os.path.join(HERE, "_imglist_query.txt"), "w").write("\n".join(qs) + "\n")
    open(os.path.join(HERE, "_imglist_gallery.txt"), "w").write("\n".join(gs) + "\n")
    nsame = sum(p["gt_same"] for p in chosen)
    print(f"selected {len(chosen)} pairs: {nsame} same / {len(chosen)-nsame} diff; "
          f"{len(qs)} uniq query, {len(gs)} uniq gallery imgs")
    from collections import Counter
    print("by n_visible:", dict(sorted(Counter(p['query_n_visible'] for p in chosen).items())))


PROMPT_A = ("Image 1 and Image 2 are cropped pedestrian images from surveillance cameras. "
            "Decide if they show the SAME person (same identity). "
            "Reply with EXACTLY one word: YES or NO.")


def prompt_b(part_vis):
    vis = [p for p in PARTS if part_vis.get(p)]
    occ = [p for p in PARTS if not part_vis.get(p)]
    vs = ", ".join(vis) if vis else "none"
    os_ = ", ".join(occ) if occ else "none"
    return ("Image 1 and Image 2 are cropped pedestrian images from surveillance cameras. "
            f"In Image 1, the VISIBLE body parts are: {vs}. The OCCLUDED/missing parts are: {os_}. "
            "Compare ONLY the mutually-visible parts; ignore occluded or missing regions. "
            "Decide if they show the SAME person (same identity). "
            "Reply with EXACTLY one word: YES or NO.")


def call_codex(qpath, gpath, prompt, effort="low", timeout=150):
    env = dict(os.environ)
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env[k] = PROXY
    env["NO_PROXY"] = env["no_proxy"] = "localhost,127.0.0.1,::1"
    try:
        p = subprocess.run([CODEX, "exec", "-s", "read-only", "-c", f"model_reasoning_effort={effort}",
                            "-i", qpath, "-i", gpath],
                           input=prompt, capture_output=True, text=True, env=env, timeout=timeout)
        out = p.stdout
    except subprocess.TimeoutExpired:
        return "UNK", "timeout"
    # parse: take the last standalone YES/NO token in output
    ans = "UNK"
    for line in out.splitlines():
        t = line.strip().upper().rstrip(".")
        if t in ("YES", "NO"):
            ans = t
    if ans == "UNK":
        u = out.upper()
        # fallback: last occurrence
        iy, ino = u.rfind("YES"), u.rfind("NO")
        if max(iy, ino) >= 0:
            ans = "YES" if iy > ino else "NO"
    return ans, out[-200:]


def run(workers=8, effort="low"):
    sub = json.load(open(SUB))
    done = {}
    if os.path.exists(RESULTS):
        for ln in open(RESULTS):
            try:
                r = json.loads(ln); done[(r["query_img"], r["gallery_img"])] = r
            except Exception:
                pass
    todo = [p for p in sub if (p["query_img"], p["gallery_img"]) not in done]
    print(f"{len(sub)} pairs, {len(done)} done, {len(todo)} todo")
    fout = open(RESULTS, "a")
    lock = __import__("threading").Lock()

    def work(p):
        q = os.path.join(IMG_Q, p["query_img"]); g = os.path.join(IMG_G, p["gallery_img"])
        if not (os.path.exists(q) and os.path.exists(g)):
            return {"query_img": p["query_img"], "gallery_img": p["gallery_img"], "err": "missing_img"}
        a, _ = call_codex(q, g, PROMPT_A, effort)
        b, _ = call_codex(q, g, prompt_b(p["query_part_visibility"]), effort)
        return {"query_img": p["query_img"], "gallery_img": p["gallery_img"],
                "gt_same": p["gt_same"], "n_visible": p["query_n_visible"], "ans_A": a, "ans_B": b}

    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, p) for p in todo]
        for f in as_completed(futs):
            r = f.result()
            with lock:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n"); fout.flush()
            n += 1
            if n % 20 == 0:
                print(f"  {n}/{len(todo)} done")
    fout.close()
    print("run complete")


def analyze():
    rows = [json.loads(ln) for ln in open(RESULTS) if ln.strip()]
    rows = [r for r in rows if "gt_same" in r]
    def acc(rows, key):
        ok = [( "YES" if r["gt_same"] else "NO") == r[key] for r in rows if r[key] in ("YES", "NO")]
        return (sum(ok) / len(ok) * 100, len(ok)) if ok else (float("nan"), 0)
    aA, nA = acc(rows, "ans_A"); aB, nB = acc(rows, "ans_B")
    print(f"\n=== exp323 codex (GPT-5.5) A/B on {len(rows)} heavy-occ pairs ===")
    print(f"Prompt A (naked):        acc {aA:.1f}%  (n={nA})")
    print(f"Prompt B (pose-grounded): acc {aB:.1f}%  (n={nB})")
    print(f"  delta (B - A): {aB-aA:+.1f} pp")
    # by n_visible bucket (red-line #6 defense: gain bigger at heavier occlusion)
    print("\nby n_visible bucket (occlusion level):")
    print("  n_vis | #pairs | accA | accB | B-A")
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows: by[r["n_visible"]].append(r)
    for b in sorted(by):
        a1, _ = acc(by[b], "ans_A"); a2, _ = acc(by[b], "ans_B")
        print(f"  {b:5d} | {len(by[b]):6d} | {a1:4.0f} | {a2:4.0f} | {a2-a1:+.0f}")
    # flip analysis
    Bfix = sum(1 for r in rows if r["ans_A"] != ("YES" if r["gt_same"] else "NO") and r["ans_B"] == ("YES" if r["gt_same"] else "NO"))
    Bbreak = sum(1 for r in rows if r["ans_A"] == ("YES" if r["gt_same"] else "NO") and r["ans_B"] != ("YES" if r["gt_same"] else "NO"))
    print(f"\nflips: B fixed {Bfix} that A got wrong; B broke {Bbreak} that A got right (net {Bfix-Bbreak:+d})")
    unk = sum(1 for r in rows if r["ans_A"] == "UNK" or r["ans_B"] == "UNK")
    print(f"UNK answers: {unk}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "run", "analyze"])
    ap.add_argument("--n", type=int, default=150, help="pairs per side (same/diff)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--effort", default="low")
    a = ap.parse_args()
    if a.mode == "select": select(n_per_side=a.n)
    elif a.mode == "run": run(workers=a.workers, effort=a.effort)
    elif a.mode == "analyze": analyze()
