"""FastAPI server for CaP-X interactive web UI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.request import urlopen, Request as UrlRequest

# Bypass proxy for local services (IK solver, LLM server, etc.)
_no_proxy = os.environ.get("NO_PROXY", "")
for host in ("127.0.0.1", "localhost", "::1"):
    if host not in _no_proxy:
        _no_proxy = f"{host},{_no_proxy}" if _no_proxy else host
os.environ["NO_PROXY"] = _no_proxy
os.environ["no_proxy"] = _no_proxy

import tyro
import uvicorn
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from capx.envs.configs.instantiate import instantiate
from capx.utils.launch_utils import _load_config
from capx.web.async_trial_runner import LaunchArgsCompat, run_trial_async
from capx.web.models import (
    ConfigItem,
    ConfigListResponse,
    InjectPromptCommand,
    LoadConfigRequest,
    LoadConfigResponse,
    SessionState,
    SessionStatusResponse,
    StartTrialRequest,
    StartTrialResponse,
    StateUpdateEvent,
    StopCommand,
    StopTrialRequest,
    StopTrialResponse,
)
from capx.web.session_manager import Session, get_session_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text-mode helpers: parse environment descriptions → object specs
# ---------------------------------------------------------------------------
import re as _re
from typing import Any as _Any

_COLOUR_MAP_ZH: dict[str, str] = {
    "红": "red", "红色": "red",
    "绿": "green", "绿色": "green",
    "蓝": "blue", "蓝色": "blue",
    "橙": "orange", "橙色": "orange", "橘": "orange", "橘色": "orange",
    "黄": "yellow", "黄色": "yellow",
    "黑": "black", "黑色": "black",
    "白": "white", "白色": "white",
    "粉": "pink", "粉色": "pink", "粉红": "pink", "粉红色": "pink",
    "紫": "purple", "紫色": "purple",
    "灰": "gray", "灰色": "gray",
    "棕": "brown", "棕色": "brown", "褐": "brown", "褐色": "brown",
    "青": "cyan", "青色": "cyan",
}
_COLOUR_MAP_EN: dict[str, str] = {
    "red": "red", "green": "green", "blue": "blue",
    "orange": "orange", "yellow": "yellow", "black": "black", "white": "white",
    "pink": "pink", "purple": "purple", "gray": "gray", "grey": "gray",
    "brown": "brown", "cyan": "cyan",
}

_ZH_DIGITS: dict[str, int] = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

# Object type vocabularies (Chinese / English → canonical type key)
_OBJECT_TYPE_MAP_ZH: dict[str, str] = {
    "方块": "box", "积木": "box", "立方体": "box", "盒子": "box",
    "球": "ball", "球体": "ball",
    "圆柱": "cylinder", "柱体": "cylinder", "柱子": "cylinder",
    "瓶子": "bottle", "瓶": "bottle",
    "易拉罐": "can", "罐子": "can", "罐": "can",
    "牛奶盒": "milk", "牛奶": "milk",
    "面包": "bread",
    "麦片盒": "cereal", "麦片": "cereal",
    "锥体": "cone", "圆锥": "cone",
    "柠檬": "lemon",
}
_OBJECT_TYPE_MAP_EN: dict[str, str] = {
    "cube": "box", "cubes": "box", "box": "box", "boxes": "box",
    "block": "box", "blocks": "box",
    "ball": "ball", "balls": "ball", "sphere": "ball", "spheres": "ball",
    "cylinder": "cylinder", "cylinders": "cylinder",
    "bottle": "bottle", "bottles": "bottle",
    "can": "can", "cans": "can",
    "cone": "cone", "cones": "cone",
}

# Types that accept colour/material
_COLOURABLE_TYPES = {"box", "ball", "cylinder", "cone"}

# Human-readable type names (for prompt generation)
_TYPE_DISPLAY: dict[str, str] = {
    "box": "cube", "ball": "ball", "cylinder": "cylinder",
    "cone": "cone", "bottle": "bottle", "can": "can",
    "milk": "milk carton", "bread": "bread", "cereal": "cereal box",
    "lemon": "lemon",
}

_ZH_COLOUR_RE = r"(红色?|绿色?|蓝色?|橙色?|橘色?|黄色?|黑色?|白色?|粉红色?|粉色?|粉|紫色?|灰色?|棕色?|褐色?|青色?)"
# Build regex for ZH object types (longest first to avoid partial match)
_ZH_OBJ_TYPES_SORTED = sorted(_OBJECT_TYPE_MAP_ZH.keys(), key=len, reverse=True)
_ZH_OBJ_RE = r"(" + "|".join(_re.escape(t) for t in _ZH_OBJ_TYPES_SORTED) + r")"


def _zh_to_int(raw: str) -> int:
    """Convert a Chinese or Arabic numeral string to int."""
    return _ZH_DIGITS.get(raw, None) or int(raw)


def _parse_object_specs(user_instruction: str) -> list[dict[str, _Any]]:
    """Extract object type/colour/count specs from Chinese or English text.

    Returns a list of dicts with keys ``name``, ``type``, and optionally
    ``colour``.  Returns an empty list if nothing recognised.
    """
    # (count, type_key, colour_or_None)
    specs: list[tuple[int, str, str | None]] = []
    _num = r"(\d+|[一二两三四五六七八九十])"

    # ── Pattern A: "红色，绿色，橙色各三个[球/方块/...]" ──────────────
    ge_pat = _re.compile(
        r"(?:" + _ZH_COLOUR_RE + r"[，,、和及]\s*)+"
        + _ZH_COLOUR_RE
        + r"\s*各\s*" + _num + r"\s*(?:个|块)"
        + r"(?:\s*" + _ZH_OBJ_RE + r")?",
    )
    for m in ge_pat.finditer(user_instruction):
        count = _zh_to_int(m.group(m.lastindex - 1) if m.lastindex and m.group(m.lastindex) in _OBJECT_TYPE_MAP_ZH else m.group(m.lastindex))
        # Detect object type from trailing noun or default to box
        span_text = m.group(0)
        obj_type = "box"
        for ot in _ZH_OBJ_TYPES_SORTED:
            if ot in span_text:
                obj_type = _OBJECT_TYPE_MAP_ZH[ot]
                break
        # Find count — it's after 各
        count_m = _re.search(r"各\s*" + _num, span_text)
        if count_m:
            count = _zh_to_int(count_m.group(1))
        for cm in _re.finditer(_ZH_COLOUR_RE, span_text.split("各")[0]):
            colour = _COLOUR_MAP_ZH.get(cm.group(1))
            if colour:
                specs.append((count, obj_type, colour))

    # ── Pattern A2: "瓶子、锥体、积木各五个" (object types + 各 + count) ─
    ge_obj_pat = _re.compile(
        r"(?:" + _ZH_OBJ_RE + r"[，,、和及]\s*)+"
        + _ZH_OBJ_RE
        + r"\s*各\s*" + _num + r"\s*个",
    )
    for m in ge_obj_pat.finditer(user_instruction):
        span_text = m.group(0)
        count_m = _re.search(r"各\s*" + _num, span_text)
        count = _zh_to_int(count_m.group(1)) if count_m else 1
        for om in _re.finditer(_ZH_OBJ_RE, span_text.split("各")[0]):
            obj_type = _OBJECT_TYPE_MAP_ZH.get(om.group(1), "box")
            if not any(t == obj_type for _, t, _ in specs):
                specs.append((count, obj_type, None))

    # ── Pattern B: "3个红色方块/球/圆柱/瓶子" ────────────────────────
    zh_pat = _re.compile(
        _num + r"\s*个\s*" + _ZH_COLOUR_RE + r"\s*"
        r"(?:的\s*)?" + _ZH_OBJ_RE,
        _re.IGNORECASE,
    )
    for m in zh_pat.finditer(user_instruction):
        count = _zh_to_int(m.group(1))
        colour = _COLOUR_MAP_ZH.get(m.group(2))
        obj_type = _OBJECT_TYPE_MAP_ZH.get(m.group(3), "box")
        key = (obj_type, colour)
        if key not in {(t, c) for _, t, c in specs}:
            specs.append((count, obj_type, colour))

    # ── Pattern B2: "3个瓶子", "10个多种颜色的瓶子" (no specific colour) ─
    zh_pat_nocolour = _re.compile(
        _num + r"\s*个\s*(?:[^\d一二两三四五六七八九十]{0,8}?)" + _ZH_OBJ_RE,
    )
    for m in zh_pat_nocolour.finditer(user_instruction):
        count = _zh_to_int(m.group(1))
        obj_type = _OBJECT_TYPE_MAP_ZH.get(m.group(2), "box")
        if not any(t == obj_type and c is None for _, t, c in specs):
            specs.append((count, obj_type, None))

    # ── Pattern C: "N个颜色" without object noun (方块/积木 elsewhere) ─
    if not specs:
        has_obj_word = any(w in user_instruction for w in _OBJECT_TYPE_MAP_ZH)
        if has_obj_word:
            zh_loose = _re.compile(_num + r"\s*个\s*" + _ZH_COLOUR_RE)
            # Determine default type from context
            default_type = "box"
            for ot in _ZH_OBJ_TYPES_SORTED:
                if ot in user_instruction:
                    default_type = _OBJECT_TYPE_MAP_ZH[ot]
                    break
            for m in zh_loose.finditer(user_instruction):
                colour = _COLOUR_MAP_ZH.get(m.group(2))
                key = (default_type, colour)
                if key not in {(t, c) for _, t, c in specs}:
                    specs.append((_zh_to_int(m.group(1)), default_type, colour))

    # ── Pattern D: "一个黑色的积木/球/瓶子" ──────────────────────────
    zh_pat_de = _re.compile(
        _num + r"\s*个\s*" + _ZH_COLOUR_RE + r"\s*的\s*" + _ZH_OBJ_RE,
        _re.IGNORECASE,
    )
    for m in zh_pat_de.finditer(user_instruction):
        colour = _COLOUR_MAP_ZH.get(m.group(2))
        obj_type = _OBJECT_TYPE_MAP_ZH.get(m.group(3), "box")
        key = (obj_type, colour)
        if key not in {(t, c) for _, t, c in specs}:
            specs.append((_zh_to_int(m.group(1)), obj_type, colour))

    # ── English: "3 red cubes", "2 blue balls", "1 bottle" ──────────
    en_colour_re = r"(red|green|blue|orange|yellow|black|white)"
    en_obj_re = r"(cubes?|boxes?|blocks?|balls?|spheres?|cylinders?|bottles?|cans?|cones?)"
    en_pat = _re.compile(r"(\d+)\s+" + en_colour_re + r"\s+" + en_obj_re, _re.IGNORECASE)
    for m in en_pat.finditer(user_instruction):
        colour = _COLOUR_MAP_EN.get(m.group(2).lower())
        obj_type = _OBJECT_TYPE_MAP_EN.get(m.group(3).lower(), "box")
        key = (obj_type, colour)
        if key not in {(t, c) for _, t, c in specs}:
            specs.append((int(m.group(1)), obj_type, colour))

    # English no-colour: "3 bottles", "1 can"
    en_pat_nc = _re.compile(r"(\d+)\s+" + en_obj_re, _re.IGNORECASE)
    for m in en_pat_nc.finditer(user_instruction):
        obj_type = _OBJECT_TYPE_MAP_EN.get(m.group(2).lower(), "box")
        if not any(t == obj_type for _, t, _ in specs):
            specs.append((int(m.group(1)), obj_type, None))

    if not specs:
        return []

    # Detect "多种颜色" / "多色" / "不同颜色" / "不同的颜色" → auto-assign colours
    _multi_colour = bool(_re.search(r"多[种]?[颜]?色|不同[的]?[颜]?色|各[种]?颜色", user_instruction))
    _COLOUR_CYCLE = ["red", "green", "blue", "orange", "yellow", "black",
                     "white", "pink", "purple", "gray", "brown", "cyan"]

    # Build named object list
    _NON_COLOURABLE = {"bottle", "can", "milk", "bread", "cereal", "lemon"}
    result: list[dict[str, _Any]] = []
    type_colour_counters: dict[str, int] = {}
    for count, obj_type, colour in specs:
        for i in range(min(count, 10)):
            actual_colour = colour
            if actual_colour is None and _multi_colour:
                actual_colour = _COLOUR_CYCLE[i % len(_COLOUR_CYCLE)]
            # XML objects have fixed appearance — don't assign colour
            if obj_type in _NON_COLOURABLE:
                actual_colour = None
            base = f"{actual_colour}_{obj_type}" if actual_colour else obj_type
            type_colour_counters[base] = type_colour_counters.get(base, 0) + 1
            idx = type_colour_counters[base]
            name = f"{base}_{idx}"
            spec_dict: dict[str, _Any] = {"name": name, "type": obj_type}
            if actual_colour:
                spec_dict["colour"] = actual_colour
            result.append(spec_dict)

    return result[:30]


_MULTI_TURN_PROMPT = """\
The previously generated code has been executed. Here are the results:

{executed_code}

Console stdout:
{console_stdout}

Console stderr:
{console_stderr}

Analyze the execution results and the grasp history (appended below if any grasps were attempted). \
If grasps FAILED (object stayed on table while gripper lifted), adjust your strategy:
- Shift grasp XY position slightly toward the object center
- Lower z_approach for a more precise descent (e.g. 0.05 instead of 0.1)
- Use pick_object(object_name) which auto-retries with random XY offsets
- Ensure the gripper is fully open before approaching

If there were errors or the task was not completed correctly, generate corrected Python code. \
Only use objects listed in the original prompt — do NOT create or reference extra objects."""


_LLM_PARSE_PROMPT = """\
You are a structured data extractor. Given a user's environment description, output a JSON array of object specs.

Each object is a dict with keys:
- "name": str — format: {colour}_{type}_{index} (e.g. "red_box_1", "bottle_2")
- "type": str — one of: box, ball, cylinder, cone, bottle, can, milk, bread, cereal, lemon
- "colour": str (optional) — one of: red, green, blue, orange, yellow, black, white, pink, purple, gray, brown, cyan

Rules:
- "box" = cube/block/积木/方块, "ball" = sphere/球, "cylinder" = 柱体/圆柱, "cone" = 锥体/圆锥
- "bottle/can/milk/bread/cereal/lemon" are fixed-mesh objects with FIXED appearance — do NOT assign colour to them (they always look the same regardless). Name them as type_index (e.g. "bottle_1", "can_2")
- Only box, ball, cylinder, cone support custom colours
- If the user says "多种颜色" or "不同颜色" or "various colours", assign different colours ONLY to colourable types (box/ball/cylinder/cone)
- Maximum 30 objects total
- Output ONLY the JSON array, no explanation, no markdown fences

User's environment description:
{instruction}"""


def _llm_parse_object_specs(
    user_instruction: str, server_url: str, model: str
) -> list[dict[str, _Any]]:
    """Use an LLM to parse environment description into object specs.

    Falls back to empty list on any failure.
    """
    import requests as _requests

    prompt_text = _LLM_PARSE_PROMPT.format(instruction=user_instruction)
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 2048,
        "messages": [
            {"role": "system", "content": "You output only valid JSON arrays."},
            {"role": "user", "content": prompt_text},
        ],
    }

    try:
        resp = _requests.post(
            server_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        content = body["choices"][0]["message"]["content"]

        # Strip possible markdown fences
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[: content.rfind("```")]
        content = content.strip()

        specs = json.loads(content)
        if not isinstance(specs, list):
            return []

        _VALID_TYPES = {"box", "ball", "cylinder", "cone", "bottle", "can",
                        "milk", "bread", "cereal", "lemon"}
        _VALID_COLOURS = {"red", "green", "blue", "orange", "yellow", "black",
                          "white", "pink", "purple", "gray", "brown", "cyan"}

        result: list[dict[str, _Any]] = []
        for s in specs[:30]:
            if not isinstance(s, dict) or "name" not in s or "type" not in s:
                continue
            if s["type"] not in _VALID_TYPES:
                continue
            clean: dict[str, _Any] = {"name": s["name"], "type": s["type"]}
            if s.get("colour") in _VALID_COLOURS:
                clean["colour"] = s["colour"]
            result.append(clean)

        return result

    except Exception as exc:
        logger.warning(f"LLM object spec parsing failed: {exc}")
        return []


def _build_multi_object_prompt(
    object_specs: list[dict[str, _Any]], user_instruction: str
) -> str:
    """Build a task prompt listing available objects."""
    lines = []
    for spec in object_specs:
        display_type = _TYPE_DISPLAY.get(spec["type"], spec["type"])
        colour = spec.get("colour", "")
        desc = f"{colour + ' ' if colour else ''}{display_type}"
        lines.append(f'- "{spec["name"]}" ({desc})')
    object_list = "\n".join(lines)

    task_part = user_instruction
    for marker in ["[任务指令]", "[Task]"]:
        if marker in user_instruction:
            task_part = user_instruction.split(marker, 1)[1].strip()
            break

    return f"""You are controlling a Franka Emika robot with the API described below.

Environment objects on the table:
{object_list}

Task: {task_part}

Rules:
- PREFER pick_object("object name") for picking — it handles retries and logs grasp results automatically.
- Use get_object_pose("object name", return_bbox_extent=True) to get the position, quaternion, and bounding box of an object.
- Use sample_grasp_pose("object name") to get a grasp pose for an object.
- Use goto_pose(position, quaternion_wxyz, z_approach=0.1) to move the gripper.
- Use open_gripper() and close_gripper() to control the gripper.
- Use is_grasping("object name") after lifting to verify grasp success.
- The extent from get_object_pose(..., return_bbox_extent=True) is the FULL side length. Use extent[2]/2 for half-height.
- For placement orientation, reuse the grasp quaternion from sample_grasp_pose. Do NOT use the quaternion from get_object_pose (it is unreliable for orientation).
- Always use z_approach=0.1 when approaching an object for grasping or placing.
- After grasping, lift the object to a safe height (at least +0.15m in Z) before moving laterally to the placement location.
- Object names must match EXACTLY as listed above (e.g. "red_box_1", not "red cube" or "box 1").

IMPORTANT: Only use the exact object names listed in "Environment objects" above. Do NOT invent or reference other object names."""


def _build_layered_prompt(
    object_specs: list[dict[str, _Any]],
    user_instruction: str,
    code_hint: str | None = None,
) -> str:
    """Build a layered prompt: raw instructions + object catalog + API toolkit.

    The local code does minimal processing — the LLM interprets user intent,
    maps it to available objects, and composes API calls from examples.
    """

    # ── Layer 1: raw user instruction, passed verbatim ─────────────────
    layer1 = f"""\
═══════════════════════════════════════════════════════════
LAYER 1 — USER INSTRUCTIONS (原始用户指令)
═══════════════════════════════════════════════════════════

{user_instruction}
"""

    # ── Layer 2: object catalog ────────────────────────────────────────
    #   2a: what's already on the table (if parsed)
    #   2b: full type/colour reference so the LLM understands the format
    if object_specs:
        inv_lines = []
        for spec in object_specs:
            display_type = _TYPE_DISPLAY.get(spec["type"], spec["type"])
            colour = spec.get("colour", "")
            label = f"{colour + ' ' if colour else ''}{display_type}"
            inv_lines.append(f'  "{spec["name"]}"  —  {label}')
        inventory = "\n".join(inv_lines)
    else:
        inventory = "  (no objects were parsed from the instruction; the default environment is used)"

    layer2 = f"""\
═══════════════════════════════════════════════════════════
LAYER 2 — ENVIRONMENT OBJECT REFERENCE (物体参考手册)
═══════════════════════════════════════════════════════════

### Objects currently on the table
{inventory}

### Supported object types catalog
Each object is described as a dict:  {{"name": str, "type": str, "colour": str (optional)}}

| type     | accepts colour? | description          | example spec                                                     |
|----------|----------------|----------------------|------------------------------------------------------------------|
| box      | yes            | cube / block         | {{"name": "red_box_1",    "type": "box",      "colour": "red"}}    |
| ball     | yes            | sphere               | {{"name": "blue_ball_1",  "type": "ball",     "colour": "blue"}}   |
| cylinder | yes            | cylinder             | {{"name": "green_cylinder_1", "type": "cylinder", "colour": "green"}} |
| cone     | yes            | cone                 | {{"name": "yellow_cone_1", "type": "cone",   "colour": "yellow"}}  |
| bottle   | no             | bottle (fixed mesh)  | {{"name": "bottle_1",     "type": "bottle"}}                       |
| can      | no             | soda can (fixed mesh)| {{"name": "can_1",        "type": "can"}}                          |
| milk     | no             | milk carton          | {{"name": "milk_1",       "type": "milk"}}                         |
| bread    | no             | bread loaf           | {{"name": "bread_1",      "type": "bread"}}                        |
| cereal   | no             | cereal box           | {{"name": "cereal_1",     "type": "cereal"}}                       |
| lemon    | no             | lemon                | {{"name": "lemon_1",      "type": "lemon"}}                        |

### Supported colours
red, green, blue, orange, yellow, black, white, pink, purple, gray, brown, cyan

### Naming convention
  colour_type_index  →  e.g. "red_box_1", "blue_ball_2", "bottle_1"
  Objects without colour: type_index  →  e.g. "bottle_1", "can_2"

### Important
- bottle/can/milk/bread/cereal/lemon have FIXED appearance (always the same colour). Do NOT try to distinguish them by colour.
- To identify objects, use their NAME (e.g. "bottle_1", "bottle_2"), not their colour.
- In your code, identify objects ONLY by name. The colour information is encoded in the name for colourable objects (e.g. "red_box_1" is red).
"""

    # ── Layer 3: API toolkit with composition patterns ─────────────────
    layer3 = """\
═══════════════════════════════════════════════════════════
LAYER 3 — API TOOLKIT (可用操作原语与组合样例)
═══════════════════════════════════════════════════════════

### Primitive functions

1. get_object_pose(object_name: str, return_bbox_extent: bool = False)
   → Returns (position, quaternion_wxyz, bbox_extent)
   - position: np.ndarray (3,) — XYZ in meters
   - quaternion_wxyz: np.ndarray (4,) — NOTE: orientation may be unreliable, prefer sample_grasp_pose for gripper orientation
   - bbox_extent: np.ndarray (3,) — full side lengths [x, y, z] in meters (only when return_bbox_extent=True)

2. sample_grasp_pose(object_name: str)
   → Returns (position, quaternion_wxyz)
   - A reliable top-down grasp pose; always use this quaternion for gripper orientation

3. goto_pose(position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0)
   → Moves gripper to target pose
   - z_approach > 0: first arrives at position + z_approach offset, then descends (for precise approach)
   - No need to call goto_pose a second time after using z_approach

4. open_gripper()  → Opens gripper fully
5. close_gripper() → Closes gripper fully

6. pick_object(object_name: str, max_attempts: int = 3) → bool
   - High-level pick-up with AUTOMATIC RETRY on grasp failure
   - Returns True if object was successfully grasped, False if all attempts failed
   - Internally: sample_grasp_pose → open → goto → close → lift → verify → retry if needed
   - **PREFERRED over manual pick sequences**

7. is_grasping(object_name: str) → bool
   - Check if the named object is currently held by the gripper
   - Call after close_gripper() and lifting to verify grasp success

8. get_grasp_history(object_name: str | None = None) → list[dict]
   - Returns past grasp attempts from the log (all or filtered by object_name)
   - Each entry has: object_name, object_type, grasp_pos, success, attempt, gripper_z_after_lift, object_z_after_lift
   - Use to analyze which objects failed and adjust strategy (e.g. different approach offsets)

### Composition patterns

**Pattern A — Pick with retry (PREFERRED):**
```python
success = pick_object("red_box_1")
if not success:
    print("Failed to pick red_box_1 after retries, skipping")
```

**Pattern A2 — Manual pick (if you need custom approach):**
```python
import numpy as np
pos, quat = sample_grasp_pose("red_box_1")
open_gripper()
goto_pose(pos, quat, z_approach=0.1)
close_gripper()
lift = pos.copy(); lift[2] += 0.15
goto_pose(lift, quat)
if not is_grasping("red_box_1"):
    open_gripper()  # failed — release and retry or skip
```

**Pattern B — Place at a location:**
```python
goto_pose(target_pos, quat, z_approach=0.1)  # descend to target
open_gripper()
lift = target_pos.copy(); lift[2] += 0.2
goto_pose(lift, quat)                        # retract upward
```

**Pattern C — Pick-and-place (full workflow):**
```python
# 1. Pick A with retry
success = pick_object("red_box_1")
if not success:
    print("Skipping red_box_1")
else:
    # 2. Compute placement on top of B
    pos_b, _, extent_b = get_object_pose("green_box_1", return_bbox_extent=True)
    place = pos_b.copy()
    place[2] += extent_b[2] / 2 + 0.02          # half-height of B + margin

    # 3. Place (use Pattern B) — use grasp quat
    _, quat = sample_grasp_pose("red_box_1")
    goto_pose(place, quat, z_approach=0.1)
    open_gripper()
```

### Key constraints
- Use pick_object() for picking — it handles retries and logs grasp results automatically
- z_approach=0.1 for all grasping/placing approaches
- Lift +0.15m in Z after grasping before any lateral movement
- Use quaternion from sample_grasp_pose for orientation (NOT from get_object_pose)
- Object names must match EXACTLY as listed in Layer 2
- ONLY operate on objects listed in Layer 2 — do NOT create, spawn, or reference objects not in the catalog
- Output ONLY executable Python code, no markdown fences

═══════════════════════════════════════════════════════════
Based on the 3 layers above, write Python code to accomplish the user's task.
═══════════════════════════════════════════════════════════
"""

    hint_section = ""
    if code_hint and code_hint.strip():
        hint_section = f"""
═══════════════════════════════════════════════════════════
ADDITIONAL CONSTRAINTS (用户额外约束)
═══════════════════════════════════════════════════════════
{code_hint.strip()}
"""

    return layer1 + "\n" + layer2 + "\n" + layer3 + hint_section


# ---------------------------------------------------------------------------
# Viser reverse-proxy helpers
# ---------------------------------------------------------------------------
_VISER_PORTS = [8080, 8081]
_viser_port_cache: int | None = None


def _find_viser_port() -> int | None:
    """Probe candidate ports to find a running Viser server (cached)."""
    global _viser_port_cache
    # Try cached port first
    if _viser_port_cache is not None:
        try:
            urlopen(f"http://localhost:{_viser_port_cache}/", timeout=1)
            return _viser_port_cache
        except Exception:
            _viser_port_cache = None
    for port in _VISER_PORTS:
        try:
            urlopen(f"http://localhost:{port}/", timeout=1)
            _viser_port_cache = port
            return port
        except Exception:
            continue
    return None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="CaP-X Interactive Web UI",
        description="Real-time interactive interface for CaP-X robot code execution",
        version="1.0.0",
    )

    # CORS for frontend dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ========================================================================
    # REST API Endpoints
    # ========================================================================

    @app.get("/api/default-config")
    async def get_default_config():
        """Return the default config path passed via launch.py (if any).

        When a config_path was provided on the CLI, ``auto_start`` is set to
        ``True`` so the frontend can kick off the trial immediately after load.
        """
        default_path = getattr(app.state, "default_config_path", None)
        return {
            "config_path": default_path,
            "auto_start": default_path is not None,
        }

    _CONFIG_LABELS: dict[str, str] = {
        "franka_robosuite_cube_stack_privileged": "方块堆叠",
        "franka_robosuite_cube_lifting_privileged": "方块抬举",
        "franka_robosuite_cube_restack_privileged": "方块重新堆叠",
        "franka_robosuite_nut_assembly_privileged": "螺母组装",
        "franka_robosuite_spill_wipe_privileged": "溢出擦拭",
        "franka_robosuite_two_arm_lift_privileged": "双臂抬举",
        "two_arm_handover_privileged": "双臂交接",
        "franka_robosuite_cube_stack_privileged_oracle": "方块堆叠 (Oracle)",
        "franka_robosuite_cube_lifting_privileged_oracle": "方块抬举 (Oracle)",
        "franka_robosuite_cube_restack_privileged_oracle": "方块重新堆叠 (Oracle)",
        "franka_robosuite_nut_assembly_privileged_oracle": "螺母组装 (Oracle)",
        "franka_robosuite_spill_wipe_privileged_oracle": "溢出擦拭 (Oracle)",
        "franka_robosuite_two_arm_lift_privileged_oracle": "双臂抬举 (Oracle)",
        "two_arm_handover_privileged_oracle": "双臂交接 (Oracle)",
    }

    @app.get("/api/configs", response_model=ConfigListResponse)
    async def list_configs():
        """List available YAML config files from all environment directories."""
        import sys as _sys
        is_macos = _sys.platform == "darwin"

        configs_root = Path("env_configs")
        if not configs_root.exists():
            return ConfigListResponse(configs=[])

        configs = []
        for yaml_file in sorted(configs_root.rglob("*.yaml")):
            if "hillclimb" in yaml_file.parts:
                continue
            path = str(yaml_file.relative_to("."))
            available = True
            reason = None

            if is_macos:
                name = yaml_file.name
                parent = yaml_file.parent.name
                if parent in ("libero", "r1pro", "real"):
                    available = False
                    reason = "需要 Linux (LIBERO/Isaac Sim/真实机器人)"
                elif "privileged" not in name:
                    available = False
                    reason = "需要 SAM3/ContactGraspNet (Linux GPU)"

            stem = yaml_file.stem
            label = _CONFIG_LABELS.get(stem)

            configs.append(ConfigItem(path=path, label=label, available=available, reason=reason))

        return ConfigListResponse(configs=configs)

    @app.post("/api/load-config", response_model=LoadConfigResponse)
    async def load_config(request: LoadConfigRequest):
        """Load and validate a YAML config file."""
        config_path = request.config_path

        if not Path(config_path).exists():
            raise HTTPException(status_code=404, detail=f"Config file not found: {config_path}")

        try:
            # Create a minimal args object for _load_config
            @dataclass
            class MinimalArgs:
                config_path: str
                server_url: str = "http://127.0.0.1:8110/chat/completions"
                model: str = "google/gemini-3.1-pro-preview"
                temperature: float = 1.0
                max_tokens: int = 20480
                reasoning_effort: str = "medium"
                api_key: str | None = None
                use_visual_feedback: bool | None = None
                use_img_differencing: bool | None = None
                visual_differencing_model: str | None = "google/gemini-3.1-pro-preview"
                visual_differencing_model_server_url: str | None = "http://127.0.0.1:8110/chat/completions"
                visual_differencing_model_api_key: str | None = None
                total_trials: int | None = None
                num_workers: int | None = None
                record_video: bool | None = None
                output_dir: str | None = None
                debug: bool = False
                use_oracle_code: bool | None = None
                use_parallel_ensemble: bool | None = None
                use_video_differencing: bool | None = None
                use_wrist_camera: bool | None = None
                use_multimodel: bool | None = None
                web_ui: bool | None = None
                web_ui_port: int | None = None

            args = MinimalArgs(config_path=config_path)
            env_factory, config, _ = await asyncio.to_thread(_load_config, args)

            # Extract task prompt (no length limit - UI handles display)
            task_prompt = env_factory.get("cfg", {}).get("prompt", "")

            return LoadConfigResponse(
                status="loaded",
                config_summary={
                    "record_video": config.get("record_video"),
                    "output_dir": config.get("output_dir"),
                    "use_visual_feedback": config.get("use_visual_feedback"),
                    "use_img_differencing": config.get("use_img_differencing"),
                    "total_trials": config.get("total_trials"),
                },
                task_prompt=task_prompt,
            )
        except Exception as e:
            logger.exception(f"Failed to load config: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/start-trial", response_model=StartTrialResponse)
    async def start_trial(request: StartTrialRequest):
        """Start a new trial execution."""
        config_path = request.config_path

        if not Path(config_path).exists():
            raise HTTPException(status_code=404, detail=f"Config file not found: {config_path}")

        manager = get_session_manager()

        # Create new session
        session = await manager.create_session()

        try:
            # Build args for config loading
            @dataclass
            class LoadArgs:
                config_path: str
                server_url: str
                model: str
                temperature: float
                max_tokens: int
                reasoning_effort: str = "medium"
                api_key: str | None = None
                use_visual_feedback: bool | None = None
                use_img_differencing: bool | None = None
                visual_differencing_model: str | None = None
                visual_differencing_model_server_url: str | None = None
                visual_differencing_model_api_key: str | None = None
                total_trials: int | None = 1
                num_workers: int | None = 1
                record_video: bool | None = None
                output_dir: str | None = None
                debug: bool = False
                use_oracle_code: bool | None = None
                use_parallel_ensemble: bool | None = None
                use_video_differencing: bool | None = None
                use_wrist_camera: bool | None = None
                use_multimodel: bool | None = None
                web_ui: bool | None = None
                web_ui_port: int | None = None

            load_args = LoadArgs(
                config_path=config_path,
                server_url=request.server_url,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                use_visual_feedback=request.use_visual_feedback,
                use_img_differencing=request.use_img_differencing,
                visual_differencing_model=request.visual_differencing_model,
                visual_differencing_model_server_url=request.visual_differencing_model_server_url,
            )
            env_factory, config, _ = await asyncio.to_thread(_load_config, load_args)

            # Process output_dir like launch.py does - add model name to path and create directory
            if config.get("output_dir"):
                from datetime import datetime
                # Add model name and timestamp to output path
                base_dir = config["output_dir"]
                model_slug = request.model.replace("/", "_")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_out_dir = f"{base_dir}/{model_slug}/{timestamp}"
                Path(new_out_dir).mkdir(parents=True, exist_ok=True)
                config["output_dir"] = new_out_dir
                logger.info(f"Output directory set to: {new_out_dir}")

            # Store in session
            session.config_path = config_path
            session.config = config
            session.env_factory = env_factory
            session.state = SessionState.LOADING_CONFIG

            # Ensure the configured API is actually registered; fall back to
            # the always-available privileged API if not.
            from capx.integrations.base_api import list_apis as _list_apis
            _registered = set(_list_apis())
            cfg_apis = session.env_factory.get("cfg", {}).get("apis", [])
            if cfg_apis and any(a not in _registered for a in cfg_apis):
                logger.warning(
                    f"Configured apis {cfg_apis} not all registered (available: {sorted(_registered)}). "
                    "Falling back to FrankaControlPrivilegedApi."
                )
                session.env_factory["cfg"]["apis"] = ["FrankaControlPrivilegedApi"]

            # Override task prompt with user instruction if provided
            # We APPEND the user instruction to the class default prompt so the LLM
            # still sees API rules, valid object names, and constraints.
            if request.user_instruction:
                logger.info(f"user_instruction received ({len(request.user_instruction)} chars): {request.user_instruction[:300]}")
                object_specs = _parse_object_specs(request.user_instruction)
                logger.info(f"Regex parsed object_specs: {object_specs}")

                # Fall back to LLM-based parsing if regex found nothing
                if not object_specs:
                    logger.info("Regex parser returned empty, trying LLM-based parsing...")
                    object_specs = await asyncio.to_thread(
                        _llm_parse_object_specs,
                        request.user_instruction,
                        request.server_url,
                        request.model,
                    )
                    logger.info(f"LLM parsed object_specs ({len(object_specs)}): {object_specs}")

                if object_specs:
                    logger.info(f"Parsed {len(object_specs)} objects from user instruction")
                    session.env_factory["_target_"] = (
                        "capx.envs.tasks.franka.franka_multi_cube.FrankaMultiCubeCodeEnv"
                    )
                    session.env_factory["cfg"]["low_level"] = {
                        "_target_": "capx.envs.simulators.robosuite_multi_objects.FrankaRobosuiteMultiObjectsLowLevel",
                        "object_specs": object_specs,
                        "privileged": True,
                        "enable_render": True,
                    }
                    session.env_factory["cfg"]["apis"] = ["FrankaControlPrivilegedApi"]
                    prompt = _build_layered_prompt(object_specs, request.user_instruction, request.code_hint)
                    session.env_factory["cfg"]["prompt"] = prompt
                    logger.info(f"Layered prompt ({len(prompt)} chars): ...{prompt[-200:]}")
                else:
                    prompt = _build_layered_prompt([], request.user_instruction, request.code_hint)
                    session.env_factory["cfg"]["prompt"] = prompt
                    logger.info(f"Layered prompt (no objects) ({len(prompt)} chars): ...{prompt[-200:]}")

                session.env_factory["cfg"]["multi_turn_prompt"] = _MULTI_TURN_PROMPT
            else:
                logger.info("No user_instruction in request")

            # Build launch args for trial runner
            trial_args = LaunchArgsCompat(
                model=request.model,
                server_url=request.server_url,
                api_key=None,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                reasoning_effort="medium",
                debug=False,
                visual_differencing_model=request.visual_differencing_model,
                visual_differencing_model_server_url=request.visual_differencing_model_server_url,
                visual_differencing_model_api_key=None,
            )

            # Set the initial settings on session (can be changed during trial)
            session.await_user_input_each_turn = request.await_user_input_each_turn
            session.execution_timeout = request.execution_timeout

            # Start trial in background task
            session.task = asyncio.create_task(
                run_trial_async(
                    session=session,
                    args=trial_args,
                )
            )

            return StartTrialResponse(
                session_id=session.session_id,
                status="started",
            )

        except Exception as e:
            logger.exception(f"Failed to start trial: {e}")
            await manager.remove_session(session.session_id)
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/stop", response_model=StopTrialResponse)
    async def stop_trial(request: StopTrialRequest):
        """Stop a running trial."""
        manager = get_session_manager()
        success = await manager.stop_session(request.session_id)

        if not success:
            raise HTTPException(status_code=404, detail="Session not found or not running")

        return StopTrialResponse(status="stopped")

    @app.post("/api/inject-prompt")
    async def inject_prompt(session_id: str, text: str):
        """Inject user prompt text into a running session."""
        manager = get_session_manager()
        success = await manager.inject_prompt(session_id, text)

        if not success:
            raise HTTPException(status_code=400, detail="Cannot inject: session not awaiting input")

        return {"status": "injected"}

    @app.get("/api/session/{session_id}", response_model=SessionStatusResponse)
    async def get_session_status(session_id: str):
        """Get the status of a session."""
        manager = get_session_manager()
        session = await manager.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return SessionStatusResponse(
            session_id=session.session_id,
            state=session.state,
            current_block_index=session.current_block_index,
            total_code_blocks=session.total_code_blocks,
            num_regenerations=session.num_regenerations,
        )

    @app.get("/api/sessions")
    async def list_sessions():
        """List all active sessions."""
        manager = get_session_manager()
        return {"sessions": manager.list_sessions()}

    @app.get("/api/active-session")
    async def get_active_session():
        """Get the currently active session (if any).

        Useful for reconnecting after page refresh.
        """
        manager = get_session_manager()
        session = manager.get_active_session()
        if session:
            return {
                "session_id": session.session_id,
                "state": session.state.value,
                "config_path": session.config_path,
            }
        return {"session_id": None}

    @app.get("/api/video/{path:path}")
    async def serve_video(path: str):
        """Serve a trial video file from the outputs directory."""
        video_path = Path("outputs") / path
        if not video_path.exists() or not video_path.suffix == ".mp4":
            raise HTTPException(404, "Video not found")
        return FileResponse(str(video_path), media_type="video/mp4")

    # ========================================================================
    # WebSocket Endpoint
    # ========================================================================

    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        """WebSocket connection for real-time trial updates."""
        manager = get_session_manager()
        session = await manager.get_session(session_id)

        if not session:
            await websocket.close(code=4004, reason="Session not found")
            return

        await websocket.accept()
        session.websockets.append(websocket)
        logger.info(f"WebSocket connected for session {session_id}")

        # Replay event history so reconnecting clients see all past messages
        if session.event_history:
            logger.info(f"Replaying {len(session.event_history)} events for session {session_id}")
            for event_json in session.event_history:
                try:
                    await websocket.send_text(event_json)
                except Exception:
                    break

        # Send current state
        await websocket.send_text(
            StateUpdateEvent(
                session_id=session_id,
                state=session.state,
            ).model_dump_json()
        )

        try:
            while True:
                # Receive commands from client
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    msg_type = message.get("type")

                    if msg_type == "stop":
                        logger.info(f"Stop command received for session {session_id}")
                        await manager.stop_session(session_id)

                    elif msg_type == "inject_prompt":
                        text = message.get("text", "")
                        if text:
                            await manager.inject_prompt(session_id, text)
                            logger.info(f"Injected prompt: {text[:50]}...")

                    elif msg_type == "resume":
                        # Put empty string to unblock the queue wait
                        if session.state == SessionState.AWAITING_USER_INPUT:
                            await session.user_injection_queue.put("")

                    elif msg_type == "update_settings":
                        # Update session settings dynamically during a trial
                        if "await_user_input_each_turn" in message:
                            session.await_user_input_each_turn = message["await_user_input_each_turn"]
                            logger.info(f"Updated await_user_input_each_turn to {session.await_user_input_each_turn}")

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {data}")

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for session {session_id}")
        finally:
            if websocket in session.websockets:
                session.websockets.remove(websocket)
            # Check if session should be cleaned up
            await manager.on_websocket_disconnect(session_id)

    # ========================================================================
    # Viser reverse proxy  (avoids port-forwarding issues)
    # ========================================================================

    async def _proxy_viser_http(path: str = "", query: str = "") -> Response:
        """Forward an HTTP request to the local Viser server."""
        port = await asyncio.to_thread(_find_viser_port)
        if port is None:
            return Response(
                content="Viser server not available — is the trial running?",
                status_code=503,
            )
        url = f"http://localhost:{port}/{path}"
        if query:
            url += f"?{query}"
        try:
            req = UrlRequest(url)
            resp = await asyncio.to_thread(urlopen, req, None, 5)
            content = await asyncio.to_thread(resp.read)
            content_type = resp.headers.get(
                "Content-Type", "application/octet-stream"
            )
            return Response(content=content, media_type=content_type)
        except Exception as exc:
            return Response(content=f"Proxy error: {exc}", status_code=502)

    @app.api_route("/viser-proxy", methods=["GET", "HEAD"])
    async def proxy_viser_root(request: Request):
        """Proxy the Viser root page (no trailing slash)."""
        return await _proxy_viser_http("", str(request.url.query))

    @app.api_route("/viser-proxy/{path:path}", methods=["GET", "HEAD"])
    async def proxy_viser_path(request: Request, path: str):
        """Proxy Viser sub-paths (assets, hdri, etc.)."""
        return await _proxy_viser_http(path, str(request.url.query))

    @app.websocket("/viser-proxy")
    async def proxy_viser_ws(websocket: WebSocket):
        """Reverse-proxy WebSocket connections to the local Viser server.

        The Viser client constructs the WS URL from ``window.location.href``
        (replacing http→ws and stripping the trailing slash), so the iframe at
        ``/viser-proxy/`` connects to ``ws://host/viser-proxy``.
        """
        port = await asyncio.to_thread(_find_viser_port)
        if port is None:
            await websocket.close(code=1013, reason="Viser not running")
            return

        # Forward subprotocols (Viser expects "viser-v<VERSION>")
        proto_header = websocket.headers.get("sec-websocket-protocol", "")
        client_protocols = [
            s.strip() for s in proto_header.split(",") if s.strip()
        ]

        import websockets
        from websockets.asyncio.client import connect as ws_connect

        try:
            viser_ws = await ws_connect(
                f"ws://localhost:{port}/",
                subprotocols=(
                    [websockets.Subprotocol(p) for p in client_protocols]
                    if client_protocols
                    else None
                ),
                compression=None,
                max_size=2**24,  # 16 MiB — Viser can send large scene blobs
            )
        except Exception as exc:
            logger.warning(f"Could not connect to Viser WS: {exc}")
            await websocket.close(code=1013, reason=str(exc))
            return

        # Accept the browser connection with Viser's selected subprotocol
        await websocket.accept(subprotocol=viser_ws.subprotocol)

        async def _client_to_viser() -> None:
            try:
                while True:
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if "bytes" in msg and msg["bytes"]:
                        await viser_ws.send(msg["bytes"])
                    elif "text" in msg and msg["text"]:
                        await viser_ws.send(msg["text"])
            except (WebSocketDisconnect, Exception):
                pass

        async def _viser_to_client() -> None:
            try:
                async for data in viser_ws:
                    if isinstance(data, bytes):
                        await websocket.send_bytes(data)
                    else:
                        await websocket.send_text(data)
            except Exception:
                pass

        try:
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(_client_to_viser()),
                    asyncio.create_task(_viser_to_client()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
        except Exception as exc:
            logger.warning(f"Viser WS proxy error: {exc}")
        finally:
            await viser_ws.close()
            try:
                await websocket.close()
            except Exception:
                pass

    # ========================================================================
    # Static file serving for frontend (production)
    # ========================================================================

    # Check if built frontend exists
    frontend_dist = Path(__file__).parent.parent.parent / "web-ui" / "dist"
    if frontend_dist.exists():
        app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

        @app.get("/")
        async def serve_frontend():
            return FileResponse(frontend_dist / "index.html")

        @app.get("/{path:path}")
        async def serve_frontend_routes(path: str):
            # Try to serve static file, otherwise serve index.html for SPA routing
            file_path = frontend_dist / path
            if file_path.exists() and file_path.is_file():
                return FileResponse(file_path)
            return FileResponse(frontend_dist / "index.html")

    return app


# ============================================================================
# CLI Entry Point
# ============================================================================


@dataclass
class ServerArgs:
    """Command-line arguments for the web server."""

    host: str = "0.0.0.0"
    """Host to bind the server to."""

    port: int = 8200
    """Port to run the server on."""

    reload: bool = False
    """Enable auto-reload for development."""


def main(args: ServerArgs | None = None) -> None:
    """Run the web server."""
    if args is None:
        args = tyro.cli(ServerArgs)

    app = create_app()

    logger.info(f"Starting CaP-X Interactive Web UI on http://{args.host}:{args.port}")
    logger.info("Frontend dev server should be running on http://localhost:5173")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
