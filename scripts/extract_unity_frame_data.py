"""Extract FOOTSIES fighter frame data from the Unity project into JSON.

Reads the ScriptableObject assets of fighter F00 (FighterData, ActionData,
AttackData) from a FootsiesV2 Unity checkout and writes
``footsiesgym/jax/frame_data.json``, which the JAX backend loads at import.

Numbers are kept as the exact decimal strings Unity serialized so that the
loader can convert them straight to float32 without a lossy intermediate.

Usage:
    python scripts/extract_unity_frame_data.py ~/PycharmProjects/FootsiesV2
"""

import argparse
import json
import pathlib
import re

import yaml

FIGHTER_DIR = "Assets/Fighter/F00"
OUT_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "footsiesgym/jax/frame_data.json"
)


def load_unity_yaml(path: pathlib.Path) -> dict:
    """Load a single-document Unity YAML asset with every scalar as a string."""
    text = path.read_text()
    text = re.sub(r"^%TAG.*$", "", text, flags=re.M)
    text = re.sub(r"^--- !u!.*$", "---", text, flags=re.M)
    return yaml.load(text, Loader=yaml.BaseLoader)["MonoBehaviour"]


def guid_of(path: pathlib.Path) -> str:
    meta = path.with_name(path.name + ".meta").read_text()
    return re.search(r"^guid: (\w+)", meta, flags=re.M).group(1)


def rect(r: dict) -> list[str]:
    return [r["x"], r["y"], r["width"], r["height"]]


def frames(d: dict) -> list[int]:
    return [int(d["startEndFrame"]["x"]), int(d["startEndFrame"]["y"])]


def int_list(hex_str: str) -> list[int]:
    """Decode Unity's hex-serialized List<int> (little-endian int32s)."""
    raw = bytes.fromhex(hex_str)
    return [
        int.from_bytes(raw[i : i + 4], "little", signed=True)
        for i in range(0, len(raw), 4)
    ]


def parse_action(a: dict) -> dict:
    as_list = lambda v: v if isinstance(v, list) else []  # "[]" loads as a list already
    return {
        "action_id": int(a["actionID"]),
        "name": a["actionName"],
        "type": int(a["Type"]),
        "frame_count": int(a["frameCount"]),
        # Fields missing from an asset take the C# default value.
        "is_loop": a.get("isLoop", "0") == "1",
        "loop_from_frame": int(a.get("loopFromFrame", "0")),
        "always_cancelable": a["alwaysCancelable"] == "1",
        "hitboxes": [
            {
                "frames": frames(h),
                "rect": rect(h["rect"]),
                "attack_id": int(h["attackID"]),
                "proximity": h["proximity"] == "1",
            }
            for h in as_list(a["hitboxes"])
        ],
        "hurtboxes": [
            {
                "frames": frames(h),
                "rect": rect(h["rect"]),
                "use_base_rect": h["useBaseRect"] == "1",
            }
            for h in as_list(a["hurtboxes"])
        ],
        "pushboxes": [
            {
                "frames": frames(h),
                "rect": rect(h["rect"]),
                "use_base_rect": h["useBaseRect"] == "1",
            }
            for h in as_list(a["pushboxes"])
        ],
        "movements": [
            {"frames": frames(m), "velocity_x": m["velocity_x"]}
            for m in as_list(a["movements"])
        ],
        "cancels": [
            {
                "frames": frames(c),
                "buffer": c["buffer"] == "1",
                "execute": c["execute"] == "1",
                "action_ids": int_list(c["actionID"]),
            }
            for c in as_list(a["cancels"])
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("unity_project", type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, default=OUT_PATH)
    args = parser.parse_args()

    root = args.unity_project.expanduser() / FIGHTER_DIR
    fighter = load_unity_yaml(root / "F00.asset")

    # Resolve the action container's GUID references in container order.
    action_files = {guid_of(p): p for p in (root / "Actions").glob("*.asset")}
    container = load_unity_yaml(root / "F00_ActionDataContainer.asset")
    actions = [
        parse_action(load_unity_yaml(action_files[ref["guid"]]))
        for ref in container["actions"]
    ]

    attacks = [
        {k: (v if k == "attackName" else int(v)) for k, v in atk.items()}
        for atk in load_unity_yaml(root / "F00_AttackDataContainer.asset")[
            "attackDataList"
        ]
    ]

    data = {
        "source": f"{FIGHTER_DIR} (FootsiesV2 Unity project)",
        "fighter": {
            "start_guard_health": int(fighter["startGuardHealth"]),
            "forward_move_speed": fighter["forwardMoveSpeed"],
            "backward_move_speed": fighter["backwardMoveSpeed"],
            "dash_allow_frame": int(fighter["dashAllowFrame"]),
            "special_attack_hold_frame": int(fighter["specialAttackHoldFrame"]),
            "can_cancel_on_whiff": fighter["canCancelOnWhiff"] == "1",
            "base_hurtbox_rect": rect(fighter["baseHurtBoxRect"]),
            "base_pushbox_rect": rect(fighter["basePushBoxRect"]),
        },
        "actions": actions,
        "attacks": attacks,
    }
    args.out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"Wrote {len(actions)} actions and {len(attacks)} attacks to {args.out}")


if __name__ == "__main__":
    main()
