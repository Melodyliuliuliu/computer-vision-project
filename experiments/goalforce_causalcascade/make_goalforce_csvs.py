"""
Convert the CausalCascade RQ2 subset (4 scenes, 21 factual/counterfactual pairs) into
Goal Force inference CSVs (forward / direct-force mode).

Reads only from ../../CausalCascade (never modifies it) and writes to ./inputs/.

Requirements (only needed to REGENERATE the CSVs; the outputs are committed):
    pip install bpy pandas pyarrow pillow        # bpy = Blender as a Python module

Run from the repo root:
    python experiments/goalforce_causalcascade/make_goalforce_csvs.py              # force 30 -> inputs/
    python experiments/goalforce_causalcascade/make_goalforce_csvs.py --force 100  # -> inputs_f100/
Non-default forces get their own folder and a "__f<force>" name suffix, so outputs of
different force settings never collide in Goal Force's output directory.

------------------------------------------------------------------------------------
Goal Force input conventions (from goal-force/src/goal_force/unified_dataset.py,
class ControlSignalDataset_Balls, which inference_goal_force.py uses):
  * The image is loaded from <csv_dir>/images/<image> and RESIZED (stretched, no crop)
    to 832x480. CausalCascade frames are 854x480, so x is squeezed by 832/854.
  * Force blob: x = coordx / width, y = coordy / height, and y is measured from the
    BOTTOM of the image (blob row = (1 - y) * 480).
  * Mass blob: uses raw int(coordx), 480 - int(coordy) directly on the 832x480 map,
    so width/height MUST be 832/480 and coords must be in that space.
  * Angle in degrees, 0 = +x (right), 90 = up in the image.
  * Inference hard-codes force range [30, 400] and mass range [1, 4]; -1 = "not given".
  * projectile_force_magnitude and target_indirect_force_magnitude must not BOTH be -1
    (the code then draws a garbage indirect blob and crashes when saving the overlay).

Design choices for this experiment (edit FORCE_MAGNITUDE / mass encoding below):
  * Forward mode: a direct force on the trigger object of the event graph (event e01),
    target_indirect_force = -1 (masked).
  * Force direction = from trigger centre toward the first downstream participant
    (event e02), computed on the FACTUAL scene and kept identical for the
    counterfactual, so the only difference within a pair is the intervention.
  * Force application point = trigger centre in the scene of that row (it moves only
    if the intervention moved/scaled the trigger itself).
  * delete / move / scale pairs: masses = -1 (mass channel empty). The factual row is
    identical for all such pairs of a scene, so it is generated once per scene.
  * mass pairs: the mass blob is put on the edited object (projectile slot if it is the
    trigger, otherwise target slot). Real mass ratios (0.015x ... 50x) do not fit
    Goal Force's [1, 4] range, so the encoding keeps the DIRECTION with maximum
    contrast: heavier -> factual 1.0, counterfactual 4.0; lighter -> factual 4.0,
    counterfactual 1.0. Real kg values are kept in manifest.csv.
  * Prompts: the official matched Level 1 prompts from tables/rq2_pairs.parquet.
"""
import argparse, json, math, os, shutil, sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CC = REPO / "CausalCascade"
OUT = Path(__file__).resolve().parent / "inputs"

SCENES = [
    "Domino_Series/l_turn_domino_obstacle_course",
    "Domino_Series/straight_domino_line",
    "Instability_Series/ball_drop_stack_collapse",
    "Instability_Series/simple_platform_collapse",
]
GF_W, GF_H = 832, 480
FORCE_MAGNITUDE = 30.0        # default: weakest force seen in Goal Force training = a "nudge"
SUFFIX = ""                   # set by --force for non-default forces
MASS_LIGHT, MASS_HEAVY = 1.0, 4.0


# ----------------------------------------------------------------------------- geometry
def project_scene(blend_path, edit):
    """Open the .blend, apply one intervention, return {object: (x, y_from_bottom)}
    in Goal Force's 832x480 space, plus frame bounds check info."""
    import bpy, mathutils
    from bpy_extras.object_utils import world_to_camera_view

    bpy.ops.wm.open_mainfile(filepath=str(blend_path), load_ui=False, use_scripts=False)
    sc = bpy.context.scene
    sc.frame_set(sc.frame_start)

    if edit is not None:
        if isinstance(edit, list):                         # JSON-patch style
            for op in edit:
                parts = op["path"].strip("/").split("/")
                if op["op"] == "remove" and len(parts) == 2:
                    bpy.data.objects.remove(bpy.data.objects[parts[1]], do_unlink=True)
                # mass replace: no geometric change
        else:
            o = bpy.data.objects[edit["object_key"]]
            if "new_scale" in edit:
                o.scale = edit["new_scale"]
            if "new_location" in edit:
                o.location = edit["new_location"]
        bpy.context.view_layer.update()

    cam = sc.camera
    out = {}
    for o in sc.objects:
        if o.type != "MESH":
            continue
        bb = [o.matrix_world @ mathutils.Vector(v) for v in o.bound_box]
        ctr = sum(bb, mathutils.Vector()) / 8
        p = world_to_camera_view(sc, cam, ctr)             # normalized, y up
        out[o.name] = (p.x * GF_W, p.y * GF_H)
    return out


def centroid(geom, keys):
    pts = [geom[k] for k in keys if k in geom]
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def angle_deg(a, b):
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


# ----------------------------------------------------------------------------- helpers
def scene_roles(scene):
    g = json.load(open(CC / "annotations/event_graphs" / f"{scene}.json"))
    reg = {p["participant_id"]: p["metadata_keys"] for p in g["participant_registry"]}
    ev = {e["event_id"]: e for e in g["events"]}
    trig_pid = ev["e01"]["participants"][0]
    nxt = [p for p in ev["e02"]["participants"] if p != trig_pid]
    return reg[trig_pid][0], [k for p in nxt for k in reg[p]]


def gf_row(image, caption, prj_xy, angle, prj_mass, tgt_xy, tgt_mass):
    return {
        "image": image,
        "projectile_force_angle": round(angle, 1),
        "projectile_force_magnitude": FORCE_MAGNITUDE,
        "projectile_coordx": int(round(prj_xy[0])),
        "projectile_coordy": int(round(prj_xy[1])),
        "projectile_mass": prj_mass,
        "target_indirect_force_angle": -1.0,
        "target_indirect_force_magnitude": -1.0,
        "target_coordx": int(round(tgt_xy[0])),
        "target_coordy": int(round(tgt_xy[1])),
        "target_mass": tgt_mass,
        "width": GF_W,
        "height": GF_H,
        "caption": caption,
    }


def write_sample(name, src_png, row):
    """One CSV per generation (like goal-force/datasets/examples), images/ next to it."""
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    # CausalCascade frames are palette PNGs (PIL mode "P"). Goal Force's loader applies
    # ToTensor() without .convert("RGB"), which turns a palette image into a 1-channel
    # map of palette INDICES (looks like a wrong grayscale). Save as true RGB instead.
    from PIL import Image
    Image.open(src_png).convert("RGB").save(OUT / "images" / f"{name}.png")
    row = dict(row, image=f"{name}.png")
    pd.DataFrame([row]).to_csv(OUT / f"{name}.csv", index=False)
    return f"{name}.csv"


def _name(n):
    return f"{n}{SUFFIX}"


# ----------------------------------------------------------------------------- main
def main():
    global OUT, FORCE_MAGNITUDE, SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", type=float, default=30.0,
                    help="direct-force magnitude (Goal Force range 30-400)")
    a = ap.parse_args()
    FORCE_MAGNITUDE = a.force
    if a.force != 30.0:
        SUFFIX = f"__f{a.force:g}"
        OUT = OUT.parent / f"inputs_f{a.force:g}"
    # remove only what this script generates (keep object_boxes.json, previews/, ...)
    OUT.mkdir(parents=True, exist_ok=True)
    for f in [*OUT.glob("*.csv"), OUT / "all_csvs.txt"]:
        f.unlink(missing_ok=True)
    shutil.rmtree(OUT / "images", ignore_errors=True)

    pairs = pd.read_parquet(
        CC / "tables/rq2_pairs.parquet",
        columns=["pair_id", "scene_id", "variant_id", "edit_type",
                 "factual_initial_frame_path", "counterfactual_initial_frame_path",
                 "factual_reference_video_path", "counterfactual_reference_video_path",
                 "factual_prompt_level1", "counterfactual_prompt_level1", "delta_json_path"],
    )
    pairs = pairs[pairs.scene_id.isin(SCENES)].sort_values("pair_id").reset_index(drop=True)
    assert len(pairs) == 21, len(pairs)

    manifest, done_factual = [], {}
    for scene in SCENES:
        blend = CC / "source_scenes/blender" / f"{scene}.blend"
        trig, nxt = scene_roles(scene)
        g_fact = project_scene(blend, None)
        ang = angle_deg(g_fact[trig], centroid(g_fact, nxt))
        sname = scene.split("/")[1]
        print(f"{scene}: trigger={trig} next={nxt} angle={ang:.1f}")

        for _, r in pairs[pairs.scene_id == scene].iterrows():
            edit = json.load(open(CC / r.delta_json_path))
            eid = r.variant_id.split("/")[-1]
            tag = f"{sname}__{r.edit_type}__{eid}"
            g_cf = g_fact if r.edit_type == "mass" else project_scene(blend, edit)
            prj_f, prj_c = g_fact[trig], g_cf[trig]

            if r.edit_type == "mass":
                obj = edit[0]["path"].split("/")[2]
                new_kg = float(edit[0]["value"])
                meta = json.load(open(CC / "annotations/scene_metadata" / f"{scene}.json"))
                old_kg = float(meta["objects"][obj]["rigid_body"]["mass"])
                heavier = new_kg > old_kg
                m_f, m_c = (MASS_LIGHT, MASS_HEAVY) if heavier else (MASS_HEAVY, MASS_LIGHT)
                if obj == trig:
                    rf = gf_row("", r.factual_prompt_level1, prj_f, ang, m_f, (0, 0), -1)
                    rc = gf_row("", r.counterfactual_prompt_level1, prj_c, ang, m_c, (0, 0), -1)
                else:
                    rf = gf_row("", r.factual_prompt_level1, prj_f, ang, -1, g_fact[obj], m_f)
                    rc = gf_row("", r.counterfactual_prompt_level1, prj_c, ang, -1, g_fact[obj], m_c)
                f_csv = write_sample(_name(f"{tag}__factual"), CC / r.factual_initial_frame_path, rf)
                mass_info = dict(mass_object=obj, mass_kg_factual=old_kg, mass_kg_cf=new_kg,
                                 mass_gf_factual=m_f, mass_gf_cf=m_c)
            else:
                if scene not in done_factual:
                    rf = gf_row("", r.factual_prompt_level1, prj_f, ang, -1, (0, 0), -1)
                    done_factual[scene] = write_sample(
                        _name(f"{sname}__factual"), CC / r.factual_initial_frame_path, rf)
                f_csv = done_factual[scene]
                rc = gf_row("", r.counterfactual_prompt_level1, prj_c, ang, -1, (0, 0), -1)
                mass_info = {}
                if not (0 <= prj_c[0] < GF_W and 0 <= prj_c[1] < GF_H):
                    print(f"  WARNING {tag}: trigger outside frame {prj_c}")

            c_csv = write_sample(_name(f"{tag}__cf"), CC / r.counterfactual_initial_frame_path, rc)
            manifest.append(dict(
                pair_id=r.pair_id, scene_id=scene, edit_type=r.edit_type, edit_id=eid,
                factual_csv=f_csv, counterfactual_csv=c_csv, trigger=trig,
                force_angle=round(ang, 1), force_magnitude=FORCE_MAGNITUDE,
                factual_reference_video=r.factual_reference_video_path,
                counterfactual_reference_video=r.counterfactual_reference_video_path,
                **mass_info))

    pd.DataFrame(manifest).to_csv(OUT / "manifest.csv", index=False)
    csvs = sorted({m["factual_csv"] for m in manifest} | {m["counterfactual_csv"] for m in manifest})
    (OUT / "all_csvs.txt").write_text(
        "\n".join(f"../experiments/goalforce_causalcascade/{OUT.name}/{c}" for c in csvs) + "\n")
    print(f"{len(manifest)} pairs, {len(csvs)} generations -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
