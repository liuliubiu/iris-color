"""巩膜参考色彩归一化评估：对 0716/0721 分组数据做校正前后 L*/grade 一致性对比。

对每张图跑一次完整流水线（强制启用 sclera_normalization），从同一份检测/采样
结果分别计算「基线（未校正）」与「校正后」的 Lab 与 Grade，保证对比只反映
色彩校正本身、不受定位波动影响。

按 人 / 人×设备 / 人×设备×条件 分组统计 L* 的均值/标准差/极差 与 grade 一致率。
同时汇总巩膜参考 Lab 分布，供标定 target_l。

用法（在 iris-vision 目录）：
    python scripts/eval_sclera_norm.py [--tag run1] [--save-overlays] [--target-l 75]
        [--set key=value ...]   # 覆盖 sclera_normalization 配置项，如 --set s_max=50
"""

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.color import classify_iris_color, extract_iris_lab_median  # noqa: E402
from app.services.debug_viz import build_debug_images  # noqa: E402
from app.services.grade import get_grade_boundaries, map_l_star_to_grade  # noqa: E402
from app.services.pipeline import AnalysisError, load_config, run_analysis  # noqa: E402

IMG_DIR = ROOT.parent / "img"
OUT_DIR = ROOT / "debug_output" / "_sclera_eval"
CONFIG_PATH = ROOT / "config" / "grade_thresholds.yaml"
EXTS = (".jpg", ".jpeg", ".png")


def _collect_images() -> list:
    datasets = [
        d for d in IMG_DIR.iterdir()
        if d.is_dir() and (d.name.startswith("0716") or d.name.startswith("0721"))
    ]
    paths = []
    for d in datasets:
        paths.extend(p for p in d.rglob("*") if p.suffix.lower() in EXTS and p.is_file())
    return sorted(paths, key=lambda p: str(p.relative_to(IMG_DIR)))


def _imread_unicode(path: Path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _imwrite_unicode(path: Path, image) -> None:
    ok, buf = cv2.imencode(path.suffix, image)
    if ok:
        buf.tofile(str(path))


def _is_cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _parse_meta(rel: Path) -> dict:
    """从相对路径解析 人/设备/光源/环境/组 元信息。"""
    parts = rel.parts
    person = next(
        (p for p in parts[:-1] if len(p) <= 2 and all(_is_cjk(c) for c in p)), "?"
    )
    device = "?"
    for p in parts[:-1]:
        low = p.lower()
        if "honor" in low:
            device = "honor70"
        elif "iphone" in low:
            device = "iphone"
    name = rel.stem
    light = "裂隙灯" if "裂隙灯" in name else ("补充光" if "补充光" in name else "")
    env = "暗" if re.search(r"_暗", name) else ("普通" if "普通环境" in name else "")
    m = re.search(r"组(\d+)", name)
    group = f"组{m.group(1)}" if m else ""
    condition = "_".join(x for x in (light, env, group) if x) or "-"
    return {
        "dataset": parts[0],
        "person": person,
        "device": device,
        "condition": condition,
    }


def _grade_of(l_star: float, boundaries) -> int:
    return map_l_star_to_grade(l_star, boundaries).grade


def _stats(values: list) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 2),
        "std": round(float(arr.std(ddof=0)), 2),
        "range": round(float(arr.max() - arr.min()), 2),
    }


def _grade_summary(grades: list) -> str:
    counts = Counter(grades)
    parts = [f"G{g}x{c}" for g, c in sorted(counts.items())]
    majority = max(counts.values()) / len(grades)
    return f"{'/'.join(parts)} 一致率{majority:.0%}"


def _print_group_table(title: str, records: list, key_fn) -> None:
    groups = {}
    for r in records:
        groups.setdefault(key_fn(r), []).append(r)
    print(f"\n== {title} ==")
    header = (
        f"{'分组':<28} {'n':>3}  {'基线 μL':>8} {'σL':>6} {'ΔL':>6}"
        f"  {'校正 μL':>8} {'σL':>6} {'ΔL':>6}   grade 基线 → 校正"
    )
    print(header)
    print("-" * len(header.encode('gbk', errors='replace')))
    for key in sorted(groups):
        rs = groups[key]
        if len(rs) < 2:
            continue
        base = _stats([r["base_l"] for r in rs])
        corr = _stats([r["corr_l"] for r in rs])
        base_g = _grade_summary([r["base_grade"] for r in rs])
        corr_g = _grade_summary([r["corr_grade"] for r in rs])
        print(
            f"{key:<28} {base['n']:>3}  {base['mean']:>8.2f} {base['std']:>6.2f} {base['range']:>6.2f}"
            f"  {corr['mean']:>8.2f} {corr['std']:>6.2f} {corr['range']:>6.2f}"
            f"   {base_g} → {corr_g}"
        )


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    args = list(sys.argv[1:])
    tag = "run"
    save_overlays = False
    target_l_override = None
    if "--tag" in args:
        i = args.index("--tag")
        tag = args[i + 1]
        del args[i : i + 2]
    if "--save-overlays" in args:
        save_overlays = True
        args.remove("--save-overlays")
    if "--target-l" in args:
        i = args.index("--target-l")
        target_l_override = float(args[i + 1])
        del args[i : i + 2]
    overrides = {}
    while "--set" in args:
        i = args.index("--set")
        key, _, value = args[i + 1].partition("=")
        try:
            overrides[key] = float(value)
        except ValueError:
            overrides[key] = value in ("true", "True", "1")
        del args[i : i + 2]

    config = load_config(CONFIG_PATH)
    sclera_cfg = config.setdefault("sclera_normalization", {})
    sclera_cfg["enabled"] = True
    if target_l_override is not None:
        sclera_cfg["target_l"] = target_l_override
    sclera_cfg.update(overrides)
    if overrides:
        print(f"overrides: {overrides}")
    boundaries = get_grade_boundaries(config)
    highlight_v = config.get("highlight_v_threshold", 240)
    eye_cfg = config.get("eye_closeup", {})
    color_sample_cap = int(eye_cfg.get("color_sample_cap", 20000))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images = _collect_images()
    print(
        f"tag={tag}  images={len(images)}  target_l={sclera_cfg.get('target_l')}  "
        f"norm_lum={sclera_cfg.get('normalize_luminance', True)}  "
        f"norm_chroma={sclera_cfg.get('normalize_chroma', True)}"
    )
    print("-" * 130)

    records = []
    failures = []
    for path in images:
        rel = path.relative_to(IMG_DIR)
        meta = _parse_meta(rel)
        image_bgr = _imread_unicode(path)
        if image_bgr is None:
            failures.append({"image": str(rel), "error": "decode_failed"})
            continue

        t0 = time.perf_counter()
        try:
            res = run_analysis(image_bgr, config, CONFIG_PATH, skip_quality=True)
        except AnalysisError as exc:
            failures.append({"image": str(rel), "error": exc.code})
            print(f"{str(rel):<80} FAIL {exc.code}")
            continue
        elapsed = time.perf_counter() - t0

        # 基线：同一份检测/采样，不加增益重取色
        mad_trim = (
            float(eye_cfg.get("color_trim_mad", 2.5)) if res.scope is not None else 0.0
        )
        base_lab = extract_iris_lab_median(
            res.work_image,
            res.detection.mask,
            highlight_v,
            sample_cap=color_sample_cap,
            masks=res.sampling,
            mad_trim=mad_trim,
            channel_gains=None,
        )
        base_grade = _grade_of(base_lab.L, boundaries)
        base_color = classify_iris_color(base_lab, config)

        corr_lab = res.lab
        corr_grade = res.grade.grade

        sclera = res.sclera
        record = {
            "image": str(rel),
            **meta,
            "elapsed_s": round(elapsed, 2),
            "sclera_status": res.sclera_status,
            "sclera_lab": [round(v, 2) for v in sclera.lab] if sclera and sclera.lab else None,
            "sclera_pixels": sclera.pixel_count if sclera else 0,
            "sclera_clipped_ratio": sclera.clipped_ratio if sclera else 0.0,
            "gains": [round(float(g), 4) for g in res.sclera_gains]
            if res.sclera_gains is not None else None,
            "base_l": round(base_lab.L, 2),
            "base_ab": [round(base_lab.a, 2), round(base_lab.b, 2)],
            "base_grade": base_grade,
            "base_color": base_color.code,
            "corr_l": round(corr_lab.L, 2),
            "corr_ab": [round(corr_lab.a, 2), round(corr_lab.b, 2)],
            "corr_grade": corr_grade,
            "corr_color": res.iris_color.code,
        }
        records.append(record)

        sclera_l = f"{sclera.lab[0]:6.1f}" if sclera and sclera.lab else "  --  "
        print(
            f"{str(rel):<80} base L={base_lab.L:6.2f} G{base_grade}"
            f" | corr L={corr_lab.L:6.2f} G{corr_grade}"
            f" | sclera L={sclera_l} {res.sclera_status}"
        )

        if save_overlays:
            overlays = build_debug_images(res, eye_cfg)
            stem = "__".join(list(rel.parts[:-1]) + [rel.stem])
            for key in ("02_iris_ring", "06_sclera_samples"):
                if key in overlays:
                    _imwrite_unicode(OUT_DIR / f"{stem}__{key}.jpg", overlays[key])

    print("-" * 130)
    if not records:
        print("无有效结果")
        return

    applied = [r for r in records if r["sclera_status"] == "applied"]
    status_counts = Counter(r["sclera_status"] for r in records)
    print(f"\n巩膜参考状态：{dict(status_counts)}  （应用率 {len(applied)}/{len(records)}）")

    sclera_labs = [r["sclera_lab"] for r in records if r["sclera_lab"]]
    if sclera_labs:
        arr = np.asarray(sclera_labs)
        print(
            f"巩膜参考 Lab（校正前，n={len(arr)}）："
            f"L* 中位数={np.median(arr[:, 0]):.1f}（P10={np.percentile(arr[:, 0], 10):.1f} "
            f"P90={np.percentile(arr[:, 0], 90):.1f}）  "
            f"a* 中位数={np.median(arr[:, 1]):.1f}  b* 中位数={np.median(arr[:, 2]):.1f}"
        )
        print("→ 建议 target_l 取上述 L* 中位数附近")

    _print_group_table("按 人 汇总（跨设备跨条件）", records, lambda r: f"{r['person']}")
    _print_group_table(
        "按 人 × 设备", records, lambda r: f"{r['person']}|{r['device']}"
    )
    _print_group_table(
        "按 人 × 设备 × 条件",
        records,
        lambda r: f"{r['person']}|{r['device']}|{r['condition']}",
    )

    out_json = OUT_DIR / f"summary_{tag}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {"records": records, "failures": failures, "status_counts": dict(status_counts)},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nfailures={len(failures)}  detail={out_json}")


if __name__ == "__main__":
    main()
