"""
Object-level timing analysis: a supplement to CausalCascade's motion-map metrics.

For every dynamic object (box from inputs/object_boxes.json) and every video
(generated factual / counterfactual, reference factual / counterfactual), this computes
a motion curve: the fraction of the object's box whose grayscale value differs from
frame 0 by more than DIFF_THR, after subtracting the frame's global median change
(to ignore brightness drift). From the curve:
    moved   = max(curve) >= MOVE_THR
    onset_s = first time curve >= MOVE_THR (seconds, using each video's own fps)

Per pair it then reports, generated vs reference:
    factual_moved_acc / cf_moved_acc  objects whose moved/not-moved status matches
                                      (does the cascade reach the right objects?)
    flip_recall                       objects whose status CHANGES between the reference
                                      factual and counterfactual: did the generated pair
                                      change them in the same direction?
    false_flips                       objects unchanged in the reference pair but changed
                                      in the generated pair
    factual_order_acc / cf_order_acc  pairwise onset order agreement over objects that
                                      moved in both (pairs closer than ORDER_TOL_S in
                                      the reference are skipped)

Pair metrics use only event-graph participants by default: bystander boxes (e.g. a
static vase that a falling domino passes in front of) light up from occlusion, so they
are listed in per_object.csv but excluded unless --include-bystanders.
Known limits: a box also sees anything that passes through it (a falling neighbour, a
deleted object's region being entered), and it cannot detect interpenetration.

Usage (pip install opencv-python numpy pandas matplotlib):
    python experiments/goalforce_causalcascade/object_timing.py \
        --gen-dir ~/gf_outputs --out-dir experiments/goalforce_causalcascade/results/timing
"""
import argparse, itertools, json, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_cf import find_generated  # noqa: E402

W, H = 416, 240          # analysis resolution (keeps the camera view's aspect ~16:9)
DIFF_THR = 0.08          # per-pixel grayscale change counted as "changed"
MOVE_THR = 0.10          # fraction of the box that must change to count as motion
ORDER_TOL_S = 0.10       # skip object pairs whose reference onsets are closer than this


def read_gray(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        frames.append(g.astype(np.float32) / 255.0)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return np.stack(frames), fps


def curves(path, boxes):
    v, fps = read_gray(path)
    d = np.abs(v - v[0])
    d = np.clip(d - np.median(d, axis=(1, 2), keepdims=True), 0, None)
    changed = d > DIFF_THR
    out = {}
    for name, (x0, x1, y0, y1) in boxes.items():
        X0, X1 = max(0, int(x0 * W)), min(W, int(np.ceil(x1 * W)))
        Y0, Y1 = max(0, int(y0 * H)), min(H, int(np.ceil(y1 * H)))
        if X1 <= X0 or Y1 <= Y0:
            continue                                   # box outside the frame
        out[name] = changed[:, Y0:Y1, X0:X1].mean(axis=(1, 2))
    return out, fps


def summarize(curve, fps):
    hit = np.nonzero(curve >= MOVE_THR)[0]
    return {"moved": bool(hit.size), "onset_s": float(hit[0] / fps) if hit.size else np.nan,
            "max_change": float(curve.max())}


def order_acc(gen, ref):
    objs = [o for o in ref if o in gen and ref[o]["moved"] and gen[o]["moved"]]
    agree = total = 0
    for a, b in itertools.combinations(objs, 2):
        dr = ref[a]["onset_s"] - ref[b]["onset_s"]
        if abs(dr) < ORDER_TOL_S:
            continue
        total += 1
        agree += np.sign(dr) == np.sign(gen[a]["onset_s"] - gen[b]["onset_s"])
    return (agree / total if total else np.nan), total


def plot_pair(path, title, data, part):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharey=True)
    for ax, key in zip(axes.flat, ["ref_factual", "gen_factual", "ref_cf", "gen_cf"]):
        c, fps = data[key]
        for name, y in c.items():
            ls = "-" if name in part else ":"
            ax.plot(np.arange(len(y)) / fps, y, ls, label=name)
        ax.axhline(MOVE_THR, color="k", lw=0.5)
        ax.set_title(key)
        ax.set_xlabel("seconds")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(title + "   (solid = event-graph participant, dotted = bystander)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", type=Path, required=True)
    ap.add_argument("--cc-root", type=Path, default=HERE.parents[1] / "CausalCascade")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results" / "timing")
    ap.add_argument("--manifest", type=Path, default=HERE / "inputs" / "manifest.csv",
                    help="e.g. inputs_f100/manifest.csv for another force setting")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--include-bystanders", action="store_true",
                    help="also use objects outside the event graph in the pair metrics")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    boxes = json.loads((HERE / "inputs" / "object_boxes.json").read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    obj_rows, pair_rows, skipped = [], [], []

    for _, m in man.iterrows():
        vids = {"gen_factual": find_generated(args.gen_dir, m.factual_csv),
                "gen_cf": find_generated(args.gen_dir, m.counterfactual_csv),
                "ref_factual": args.cc_root / m.factual_reference_video,
                "ref_cf": args.cc_root / m.counterfactual_reference_video}
        if any(p is None or not Path(p).is_file() or Path(p).stat().st_size < 1000
               for p in vids.values()):
            skipped.append(m.pair_id)
            continue
        sc = boxes["scenes"][m.scene_id]
        part = sc["participant"]
        bx = {"gen_factual": sc["factual"], "ref_factual": sc["factual"],
              "gen_cf": boxes["pairs"][m.pair_id], "ref_cf": boxes["pairs"][m.pair_id]}
        data, summ = {}, {}
        for k, p in vids.items():
            c, fps = curves(p, bx[k])
            data[k] = (c, fps)
            summ[k] = {o: summarize(y, fps) for o, y in c.items()}

        for o in sc["factual"]:
            row = {"pair_id": m.pair_id, "edit_type": m.edit_type, "object": o,
                   "participant": part.get(o, "bystander")}
            for k in vids:
                s = summ[k].get(o)
                row[f"{k}_moved"] = None if s is None else s["moved"]
                row[f"{k}_onset_s"] = None if s is None else round(s["onset_s"], 3)
            obj_rows.append(row)

        use = lambda o: args.include_bystanders or o in part
        S = {k: {o: v for o, v in d.items() if use(o)} for k, d in summ.items()}
        summ = S
        common = [o for o in summ["ref_factual"] if o in summ["ref_cf"] and o in summ["gen_cf"]]
        f_acc = np.mean([summ["gen_factual"][o]["moved"] == summ["ref_factual"][o]["moved"]
                         for o in summ["ref_factual"]])
        c_acc = np.mean([summ["gen_cf"][o]["moved"] == summ["ref_cf"][o]["moved"]
                         for o in summ["ref_cf"]])
        flips = [o for o in common if summ["ref_factual"][o]["moved"] != summ["ref_cf"][o]["moved"]]
        same = [o for o in common if o not in flips]
        rec = [(summ["gen_factual"][o]["moved"], summ["gen_cf"][o]["moved"]) ==
               (summ["ref_factual"][o]["moved"], summ["ref_cf"][o]["moved"]) for o in flips]
        false = [o for o in same if summ["gen_factual"][o]["moved"] != summ["gen_cf"][o]["moved"]]
        fo, fn = order_acc(summ["gen_factual"], summ["ref_factual"])
        co, cn = order_acc(summ["gen_cf"], summ["ref_cf"])
        pair_rows.append({
            "pair_id": m.pair_id, "scene_id": m.scene_id, "edit_type": m.edit_type,
            "factual_moved_acc": f_acc, "cf_moved_acc": c_acc,
            "ref_flipped_objects": ";".join(flips),
            "flip_recall": np.mean(rec) if rec else np.nan,
            "false_flips": ";".join(false),
            "factual_order_acc": fo, "factual_order_pairs": fn,
            "cf_order_acc": co, "cf_order_pairs": cn})
        if not args.no_plots:
            name = m.pair_id.replace("rq2/", "").replace("/", "__")
            plot_pair(args.out_dir / f"{name}.png", m.pair_id, data, part)
        print(f"{m.pair_id}: factual_moved_acc={f_acc:.2f} cf_moved_acc={c_acc:.2f} "
              f"flips={flips} flip_recall={pair_rows[-1]['flip_recall']}")

    pd.DataFrame(obj_rows).to_csv(args.out_dir / "per_object.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(args.out_dir / "per_pair_timing.csv", index=False)
    (args.out_dir / "skipped.json").write_text(json.dumps(skipped, indent=1))
    if skipped:
        print(f"{len(skipped)} pairs skipped (videos missing)")


if __name__ == "__main__":
    main()
