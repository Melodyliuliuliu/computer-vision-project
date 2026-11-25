"""
Project every dynamic (ACTIVE rigid-body) object of each scene into the image and save
its 2D bounding box, for the factual scene and for every counterfactual edit.
Output: inputs/object_boxes.json (used by object_timing.py, which needs no Blender).

Boxes are normalized [x0, x1, y0, y1] with y measured from the TOP, so they apply to
any resolution of the same camera view (reference 854x480 / 1920x1080, Goal Force
832x480).

Requirements: pip install bpy pandas pyarrow
Run from the repo root:
    python experiments/goalforce_causalcascade/make_object_boxes.py
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CC = HERE.parents[1] / "CausalCascade"


def boxes_for(blend, edit):
    import bpy, mathutils
    from bpy_extras.object_utils import world_to_camera_view

    bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=False, use_scripts=False)
    sc = bpy.context.scene
    sc.frame_set(sc.frame_start)
    if edit is not None:
        if isinstance(edit, list):
            for op in edit:
                parts = op["path"].strip("/").split("/")
                if op["op"] == "remove" and len(parts) == 2:
                    bpy.data.objects.remove(bpy.data.objects[parts[1]], do_unlink=True)
        else:
            o = bpy.data.objects[edit["object_key"]]
            if "new_scale" in edit:
                o.scale = edit["new_scale"]
            if "new_location" in edit:
                o.location = edit["new_location"]
        bpy.context.view_layer.update()
    out = {}
    for o in sc.objects:
        if o.type != "MESH" or o.rigid_body is None or o.rigid_body.type != "ACTIVE":
            continue
        ps = [world_to_camera_view(sc, sc.camera, o.matrix_world @ mathutils.Vector(v))
              for v in o.bound_box]
        out[o.name] = [round(min(p.x for p in ps), 4), round(max(p.x for p in ps), 4),
                       round(1 - max(p.y for p in ps), 4), round(1 - min(p.y for p in ps), 4)]
    return out


def main():
    man = pd.read_csv(HERE / "inputs" / "manifest.csv")
    result = {"scenes": {}, "pairs": {}}
    for scene in man.scene_id.unique():
        blend = CC / "source_scenes/blender" / f"{scene}.blend"
        g = json.load(open(CC / "annotations/event_graphs" / f"{scene}.json"))
        participant = {k: p["participant_id"] for p in g["participant_registry"]
                       for k in p["metadata_keys"]}
        result["scenes"][scene] = {"factual": boxes_for(blend, None), "participant": participant}
        for _, m in man[man.scene_id == scene].iterrows():
            edit = json.load(open(CC / "annotations/interventions" /
                                  m.pair_id.split("rq2/")[1] / "diff.json"))
            result["pairs"][m.pair_id] = boxes_for(blend, None if m.edit_type == "mass" else edit)
        print(scene, len(result["scenes"][scene]["factual"]), "dynamic objects")
    (HERE / "inputs" / "object_boxes.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
