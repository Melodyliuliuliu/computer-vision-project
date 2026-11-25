"""
CausalCascade RQ2 counterfactual metrics (Effect / Preserve IoU) for the Goal Force
runs on our 21-pair subset.

The metric code is CausalCascade's own (third_party/cf_effect_metrics.py, copied
unmodified, MIT). Their official driver (compute_cf_motion_metrics.py) requires all
225 pairs and one shared factual video per scene, so this script does the same
pairing and aggregation for our subset instead:

  * every video (generated and reference) -> 256x256 grayscale, running-background
    differencing, threshold 0.06 -> per-pixel motion mask per frame
  * spatial map  = pixel moved in ANY frame;  weighted map = fraction of frames moving
  * factual vs counterfactual maps -> preserved / suppressed / induced regions
  * Effect IoU   = IoU of (suppressed, induced) between generated and reference pairs
  * Preserve IoU = IoU of the preserved region
  * aggregate: mean over edits within a scene, then mean over scenes, per edit type

Because maps are resized to 256x256 and accumulated over the whole clip, the
generated 832x480 / 81-frame / 15 fps videos and the 854x480 / 100-120-frame / 24 fps
references need no manual alignment.

Usage (needs: pip install opencv-python numpy pandas):
    python experiments/goalforce_causalcascade/eval_cf.py \
        --gen-dir  <goal-force>/checkpoints/goal_force/step-3000-videos \
        --cc-root  CausalCascade \
        --out-dir  experiments/goalforce_causalcascade/results
"""
import argparse, json, statistics, sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "third_party"))
from cf_effect_metrics import MotionConfig, MotionMapCache, score_pair  # noqa: E402

PRIMARY = ["cf_spatial_effect_iou", "cf_weighted_effect_iou",
           "cf_preserve_spatial_iou", "cf_preserve_weighted_iou"]
DIAG = ["suppressed_spatial_iou_diagnostic", "induced_spatial_iou_diagnostic",
        "suppressed_weighted_iou_diagnostic", "induced_weighted_iou_diagnostic"]


def find_generated(gen_dir: Path, csv_name: str) -> Path | None:
    """Goal Force names outputs  step-<N>_<image stem>__prj_coords_...__seed_<s>.mp4"""
    stem = Path(csv_name).stem
    hits = [p for p in gen_dir.glob(f"step-*_{stem}__prj_coords_*.mp4")
            if not p.name.endswith(("-with-prompt.mp4", "-control-signal.mp4"))]
    if len(hits) > 1:
        raise RuntimeError(f"several outputs for {stem} (different seeds?): {hits}")
    return hits[0] if hits else None


def mean(xs):
    xs = list(xs)
    return statistics.fmean(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", type=Path, required=True)
    ap.add_argument("--cc-root", type=Path, default=HERE.parents[1] / "CausalCascade")
    ap.add_argument("--manifest", type=Path, default=HERE / "inputs" / "manifest.csv")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    cache = MotionMapCache(args.out_dir / "cache", MotionConfig())
    rows, missing = [], []
    for _, m in man.iterrows():
        gen_f = find_generated(args.gen_dir, m.factual_csv)
        gen_c = find_generated(args.gen_dir, m.counterfactual_csv)
        gt_f = args.cc_root / m.factual_reference_video
        gt_c = args.cc_root / m.counterfactual_reference_video
        lack = [n for n, p in (("gen_factual", gen_f), ("gen_cf", gen_c), ("gt_factual", gt_f),
                               ("gt_cf", gt_c)) if p is None or not Path(p).is_file()
                or Path(p).stat().st_size < 1000]        # < 1 kB = un-pulled LFS pointer
        if lack:
            missing.append({"pair_id": m.pair_id, "missing": lack})
            continue
        s = score_pair(cache.get(gt_f), cache.get(gt_c), cache.get(gen_f), cache.get(gen_c))
        rows.append({"pair_id": m.pair_id, "scene_id": m.scene_id, "edit_type": m.edit_type,
                     "edit_id": m.edit_id, **s,
                     "gen_factual_video": str(gen_f), "gen_cf_video": str(gen_c)})
        print(f"{m.pair_id}: effect={s['cf_spatial_effect_iou']:.3f} "
              f"preserve={s['cf_preserve_spatial_iou']:.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "per_pair.csv", index=False)

    summary = []
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["edit_type"]].append(r)
    for et, grp in sorted(by_type.items()):
        scenes = defaultdict(list)
        for r in grp:
            scenes[r["scene_id"]].append(r)
        rec = {"edit_type": et, "pairs": len(grp), "scenes": len(scenes)}
        for k in PRIMARY + DIAG:
            rec[k] = mean(mean(r[k] for r in sr) for sr in scenes.values())
        summary.append(rec)
    pd.DataFrame(summary).to_csv(args.out_dir / "summary_scene_macro.csv", index=False)
    (args.out_dir / "missing.json").write_text(json.dumps(missing, indent=2))

    if summary:
        print("\nscene-macro, per edit type:")
        print(pd.DataFrame(summary)[["edit_type", "pairs", "scenes"] + PRIMARY]
              .to_string(index=False, float_format="%.3f"))
    if missing:
        print(f"\n{len(missing)} pairs skipped (see missing.json)")


if __name__ == "__main__":
    main()
