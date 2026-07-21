"""巩膜白平衡 A/B：同人多条件 Grade / L* 一致性对比。

用法（在 iris-vision 目录）：

  # 内置合成图自检（无样张也可跑）
  python scripts/eval_sclera_wb.py --self-test

  # 对 img/ 下图片：关闭 vs 开启各跑一遍，按子目录（或 pairs JSON）汇总
  python scripts/eval_sclera_wb.py
  python scripts/eval_sclera_wb.py --pairs path/to/pairs.json

pairs.json 示例：
  {
    "subject_a": ["a/near.jpg", "a/far.jpg"],
    "subject_b": ["b/phone1.jpg", "b/phone2.jpg"]
  }
路径相对项目根下的 img/。未提供 pairs 时，以 img/ 一级子目录名为分组；
根目录散图各自成组（仅输出单图 before/after，不参与组内一致率）。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.pipeline import AnalysisError, load_config, run_analysis  # noqa: E402
from app.services.sclera_wb import (  # noqa: E402
    apply_sclera_white_balance,
    compute_von_kries_gains,
    correct_with_sclera_wb,
    sample_sclera_reference,
)

IMG_DIR = ROOT.parent / "img"
CONFIG_PATH = ROOT / "config" / "grade_thresholds.yaml"
EXTS = (".jpg", ".jpeg", ".png")


def _imread_unicode(path: Path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _run_one(image_bgr, config: dict, enabled: bool):
    cfg = copy.deepcopy(config)
    cfg.setdefault("eye_closeup", {}).setdefault("sclera_wb", {})["enabled"] = enabled
    return run_analysis(image_bgr, cfg, CONFIG_PATH, skip_quality=True)


def self_test() -> bool:
    """合成图：暖色偏 + 偏暗巩膜，校正后应接近目标近白。"""
    print("=== self-test ===")
    h = w = 400
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy, iris_r = 200.0, 200.0, 80.0
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    # 偏暗、略暖近白巩膜（低饱和，能过 v_min/s_max / 血管过滤）
    img[dist > iris_r] = (168, 175, 190)
    # 虹膜
    img[(dist <= iris_r) & (dist > iris_r * 0.35)] = (40, 70, 110)
    # 瞳孔
    img[dist <= iris_r * 0.35] = (10, 10, 10)

    ref = sample_sclera_reference(img, (cx, cy), iris_r, min_pixels=50)
    assert ref is not None and ref.pixel_count >= 50, "sclera sample failed"
    gains, _ = compute_von_kries_gains(ref.median_bgr, [0.92, 0.92, 0.90])
    corrected = apply_sclera_white_balance(img, gains)
    ref2 = sample_sclera_reference(corrected, (cx, cy), iris_r, min_pixels=50)
    assert ref2 is not None
    # 校正后巩膜应更接近目标（各通道更高且更均衡）
    med = np.array(ref2.median_bgr)
    assert med.min() > 200, f"sclera still too dark: {med}"
    assert med.max() - med.min() < 25, f"sclera channels not balanced: {med}"

    cfg = {
        "enabled": True,
        "min_pixels": 50,
        "target_srgb": [0.92, 0.92, 0.90],
    }
    out, result = correct_with_sclera_wb(img, (cx, cy), iris_r, cfg)
    assert result.applied and result.gains_bgr is not None
    assert out is not img
    print(f"  sample_ok  pixels={ref.pixel_count}  gains_bgr={[round(g, 3) for g in result.gains_bgr]}")
    print(f"  median_before={tuple(round(v, 1) for v in ref.median_bgr)}")
    print(f"  median_after ={tuple(round(v, 1) for v in ref2.median_bgr)}")
    print("  PASS")
    return True


def _default_groups(images: List[Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = defaultdict(list)
    for p in images:
        rel = p.relative_to(IMG_DIR)
        if len(rel.parts) >= 2:
            groups[rel.parts[0]].append(p)
        else:
            groups[f"_singleton_{rel.stem}"].append(p)
    return dict(groups)


def _load_pairs(path: Path) -> Dict[str, List[Path]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    groups: Dict[str, List[Path]] = {}
    for key, rels in data.items():
        paths = []
        for rel in rels:
            p = IMG_DIR / rel
            if p.is_file():
                paths.append(p)
            else:
                print(f"  warn: missing {rel}")
        if paths:
            groups[key] = paths
    return groups


def _group_stats(grades: List[int], l_stars: List[float]) -> dict:
    if not grades:
        return {"n": 0}
    g = np.array(grades, dtype=np.int32)
    l = np.array(l_stars, dtype=np.float64)
    return {
        "n": len(grades),
        "grade_unique": int(len(set(grades))),
        "grade_agree": bool(len(set(grades)) == 1),
        "grade_range": int(g.max() - g.min()),
        "L_std": float(np.std(l)) if len(l) > 1 else 0.0,
        "L_range": float(l.max() - l.min()) if len(l) else 0.0,
        "grades": grades,
        "L_stars": [round(x, 2) for x in l_stars],
    }


def evaluate_groups(groups: Dict[str, List[Path]], config: dict) -> None:
    print(f"=== A/B on {sum(len(v) for v in groups.values())} images / {len(groups)} groups ===")
    print(f"{'group':<28} {'off_agree':>9} {'on_agree':>8} {'off_Lstd':>8} {'on_Lstd':>8}  detail")
    print("-" * 100)

    multi = {k: v for k, v in groups.items() if len(v) >= 2 and not k.startswith("_singleton_")}
    if not multi:
        print("无多图分组：将逐张打印 off/on 的 L*/Grade（请用子目录或 --pairs 做一致性评估）")
        print("-" * 100)

    agree_off = agree_on = 0
    multi_n = 0
    lstd_off: List[float] = []
    lstd_on: List[float] = []
    applied_n = clamped_n = 0
    rows = []

    targets = multi if multi else groups
    for name in sorted(targets.keys()):
        paths = targets[name]
        off_g, off_l = [], []
        on_g, on_l = [], []
        details = []
        for path in paths:
            img = _imread_unicode(path)
            rel = str(path.relative_to(IMG_DIR)) if path.is_relative_to(IMG_DIR) else path.name
            if img is None:
                details.append(f"{rel}:decode_fail")
                continue
            try:
                r_off = _run_one(img, config, enabled=False)
                r_on = _run_one(img, config, enabled=True)
            except AnalysisError as exc:
                details.append(f"{rel}:{exc.code}")
                continue
            off_g.append(r_off.grade.grade)
            off_l.append(r_off.lab.L)
            on_g.append(r_on.grade.grade)
            on_l.append(r_on.lab.L)
            wb = r_on.sclera_wb
            wb_flag = "skip"
            if wb is not None:
                if wb.applied:
                    wb_flag = "ok"
                    applied_n += 1
                    if wb.gain_clamped:
                        clamped_n += 1
                        wb_flag = "clamp"
                else:
                    wb_flag = wb.reason
            details.append(
                f"{Path(rel).name}:G{r_off.grade.grade}->{r_on.grade.grade}"
                f"(L{r_off.lab.L:.1f}->{r_on.lab.L:.1f},{wb_flag})"
            )

        st_off = _group_stats(off_g, off_l)
        st_on = _group_stats(on_g, on_l)
        if st_off.get("n", 0) >= 2:
            multi_n += 1
            if st_off["grade_agree"]:
                agree_off += 1
            if st_on["grade_agree"]:
                agree_on += 1
            lstd_off.append(st_off["L_std"])
            lstd_on.append(st_on["L_std"])
            print(
                f"{name:<28} {str(st_off['grade_agree']):>9} {str(st_on['grade_agree']):>8} "
                f"{st_off['L_std']:8.2f} {st_on['L_std']:8.2f}  {'; '.join(details)}"
            )
        else:
            print(f"{name:<28} {'(single)':>9} {'':>8} {'':>8} {'':>8}  {'; '.join(details)}")
        rows.append({"group": name, "off": st_off, "on": st_on, "details": details})

    print("-" * 100)
    if multi_n:
        print(
            f"multi-groups={multi_n}  grade_agree off={agree_off}/{multi_n} "
            f"on={agree_on}/{multi_n}  "
            f"mean_L_std off={np.mean(lstd_off):.2f} on={np.mean(lstd_on):.2f}"
        )
        print(
            "判定建议：on 的 grade_agree 明显升高且 mean_L_std 下降，"
            "且人工复核无整批系统性变浅/变深后，再把 sclera_wb.enabled 设为 true 并视情况重标定 boundaries。"
        )
    print(f"wb_applied={applied_n}  gain_clamped={clamped_n}")

    out_path = ROOT / "debug_output" / "eval_sclera_wb_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="巩膜白平衡 A/B 评估")
    parser.add_argument("--self-test", action="store_true", help="仅跑合成图自检")
    parser.add_argument("--pairs", type=str, default="", help="pairs JSON 路径")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    # 先自检再跑实拍
    self_test()
    print()

    if not IMG_DIR.is_dir():
        print(f"未找到图片目录 {IMG_DIR}，跳过实拍 A/B。")
        return

    images = sorted(
        [p for p in IMG_DIR.rglob("*") if p.suffix.lower() in EXTS and p.is_file()],
        key=lambda p: str(p.relative_to(IMG_DIR)),
    )
    if not images:
        print(f"{IMG_DIR} 下无图片，跳过实拍 A/B。")
        return

    if args.pairs:
        groups = _load_pairs(Path(args.pairs))
    else:
        groups = _default_groups(images)

    config = load_config(CONFIG_PATH)
    evaluate_groups(groups, config)


if __name__ == "__main__":
    main()
