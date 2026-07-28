"""巩膜参考色彩归一化评估：对 0716/0721 分组数据做校正前后 L*/grade 一致性对比。

对每张图跑一次完整流水线（强制启用 sclera_normalization），从同一份检测/采样
结果分别计算「基线（未校正）」与「校正后」的 Lab 与 Grade，保证对比只反映
色彩校正本身、不受定位波动影响。

按 人 / 人×设备 / 人×设备×条件 分组统计 L* 的均值/标准差/极差 与 grade 一致率。
同时汇总巩膜参考 Lab 分布，供标定 target_l。

默认保留旧目录评估；传 --experiments 时以实验记录中 include_in_stats=1 的数据
为唯一基准，并排除回放基线与数据库 lstar_before 不一致的记录。

用法（在 iris-vision 目录）：
    python scripts/eval_sclera_norm.py [--tag run1] [--save-overlays] [--target-l 75]
        [--set key=value ...]   # 覆盖 sclera_normalization 配置项，如 --set s_max=50
    python scripts/eval_sclera_norm.py --experiments --tag candidate
        [--baseline-tolerance 1.0] [--set key=value ...]
"""

import json
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pymysql

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.color import classify_iris_color, extract_iris_lab_median  # noqa: E402
from app.services.debug_viz import build_debug_images  # noqa: E402
from app.services.grade import get_grade_boundaries, map_l_star_to_grade  # noqa: E402
from app.services.pipeline import AnalysisError, load_config, run_analysis  # noqa: E402
from app.services.experiment_stability import analyze_stability  # noqa: E402

IMG_DIR = ROOT.parent / "img"
OUT_DIR = ROOT / "debug_output" / "_sclera_eval"
CONFIG_PATH = ROOT / "config" / "grade_thresholds.yaml"
EXTS = (".jpg", ".jpeg", ".png")


def _load_experiment_rows(config: dict) -> list[dict]:
    """只读加载实验记录；评估脚本绝不覆盖数据库。"""
    exp_cfg = config.get("experiments") or {}
    fields = (
        "id, group_name, subgroup_name, operator, camera_device, light_device, "
        "illuminance, lstar_before, lstar_after, grade_before, grade_after, "
        "image_rel, skip_quality, manual_adjusted, include_in_stats"
    )
    query = (
        f"SELECT {fields} FROM experiment_records "
        "WHERE include_in_stats=1 AND lstar_before IS NOT NULL "
        "AND lstar_after IS NOT NULL ORDER BY id"
    )
    if exp_cfg.get("backend", "sqlite") == "mysql":
        mysql_cfg = exp_cfg.get("mysql") or {}
        conn = pymysql.connect(
            host=mysql_cfg.get("host", "127.0.0.1"),
            port=int(mysql_cfg.get("port", 3306)),
            user=mysql_cfg.get("user", "root"),
            password=str(mysql_cfg.get("password", "")),
            database=mysql_cfg.get("database", "iris_experiment"),
            charset=mysql_cfg.get("charset", "utf8mb4"),
            cursorclass=pymysql.cursors.DictCursor,
        )
        try:
            with conn.cursor() as cur:
                cur.execute(query)
                return list(cur.fetchall())
        finally:
            conn.close()

    db_path = ROOT / exp_cfg.get("db_path", "data/experiment_records.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(query).fetchall()]
    finally:
        conn.close()


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _manual_params_for_record(
    record: dict,
    truth: dict,
    labels: dict,
) -> dict | None:
    """尽量从历史真值恢复人工环带；无法可靠映射时回退自动定位。"""
    if not record.get("manual_adjusted"):
        return None
    rel = (record.get("image_rel") or "").replace("\\", "/")
    candidates = [rel]
    label = labels.get(rel) or {}
    if label.get("previous_rel"):
        candidates.append(str(label["previous_rel"]).replace("\\", "/"))
    gt = next((truth.get(key) for key in candidates if truth.get(key)), None)
    if not gt or not gt.get("iris") or not gt.get("pupil"):
        return None
    iris = gt["iris"]
    pupil = gt["pupil"]
    iris_r = float(iris["r"])
    pupil_r = float(pupil["r"])
    return {
        "center_x": float(iris["cx"]),
        "center_y": float(iris["cy"]),
        "pupil_radius": pupil_r,
        "inner_radius": max(pupil_r * 1.15, iris_r * 0.35),
        "outer_radius": iris_r * 0.85,
    }


def _sample_std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1)) if len(values) > 1 else 0.0


def _run_experiment_eval(
    config: dict,
    *,
    tag: str,
    save_overlays: bool,
    baseline_tolerance: float,
) -> None:
    """以实验记录为标准回放当前候选调色，输出同口径稳定性报告。"""
    boundaries = get_grade_boundaries(config)
    highlight_v = int(config.get("highlight_v_threshold", 240))
    eye_cfg = config.get("eye_closeup", {})
    color_sample_cap = int(eye_cfg.get("color_sample_cap", 20000))
    truth = _load_json(ROOT / "labels" / "ground_truth.json")
    labels = _load_json(ROOT / "labels" / "img_labels.json")
    source_rows = _load_experiment_rows(config)
    records: list[dict] = []
    failures: list[dict] = []

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"source=experiments tag={tag} rows={len(source_rows)} "
        f"baseline_tolerance={baseline_tolerance:.2f}"
    )
    for source in source_rows:
        rel_text = (source.get("image_rel") or "").replace("\\", "/")
        path = IMG_DIR / Path(rel_text)
        if not rel_text or not path.is_file():
            failures.append({"id": source["id"], "image": rel_text, "error": "image_not_found"})
            continue
        image_bgr = _imread_unicode(path)
        if image_bgr is None:
            failures.append({"id": source["id"], "image": rel_text, "error": "decode_failed"})
            continue
        manual = _manual_params_for_record(source, truth, labels)
        try:
            res = run_analysis(
                image_bgr,
                config,
                CONFIG_PATH,
                # 实验记录已由人工决定纳入统计；回放沿用对比生成逻辑，统一跳过质量门，
                # 避免把“当时勾选状态”混入候选调色算法比较。
                skip_quality=True,
                manual_detection=manual,
            )
        except AnalysisError as exc:
            failures.append({"id": source["id"], "image": rel_text, "error": exc.code})
            continue

        mad_trim = float(eye_cfg.get("color_trim_mad", 2.5)) if res.scope is not None else 0.0
        base_lab = extract_iris_lab_median(
            res.work_image,
            res.detection.mask,
            highlight_v,
            sample_cap=color_sample_cap,
            masks=res.sampling,
            mad_trim=mad_trim,
            channel_gains=None,
        )
        stored_l = float(source["lstar_before"])
        baseline_error = abs(base_lab.L - stored_l)
        eligible = baseline_error <= baseline_tolerance
        sclera = res.sclera
        record = {
            "id": int(source["id"]),
            "image": rel_text,
            "person": source.get("operator"),
            "device": source.get("camera_device"),
            "condition": "|".join(str(source.get(k) or "-") for k in ("light_device", "illuminance")),
            "group_name": source.get("group_name"),
            "subgroup_name": source.get("subgroup_name"),
            "operator": source.get("operator"),
            "camera_device": source.get("camera_device"),
            "light_device": source.get("light_device"),
            "illuminance": source.get("illuminance"),
            "stored_l_before": round(stored_l, 3),
            "baseline_error": round(baseline_error, 3),
            "eligible": eligible,
            "manual_truth_used": manual is not None,
            "sclera_status": res.sclera_status,
            "sclera_lab": [round(v, 3) for v in sclera.lab] if sclera and sclera.lab else None,
            "sclera_pixels": sclera.pixel_count if sclera else 0,
            "sclera_quality": sclera.quality_score if sclera else 0.0,
            "side_luminance_gap": sclera.side_luminance_gap if sclera else 0.0,
            "luminance_mad_ratio": sclera.luminance_mad_ratio if sclera else 0.0,
            "effective_luminance_strength": (
                sclera.effective_luminance_strength if sclera else 0.0
            ),
            "effective_chroma_strength": sclera.effective_chroma_strength if sclera else 0.0,
            "camera_profile": sclera.camera_profile if sclera else None,
            "requested_lstar_delta": sclera.requested_lstar_delta if sclera else 0.0,
            "gains_limited": bool(sclera.gains_limited) if sclera else False,
            "gains": [round(float(g), 5) for g in res.sclera_gains]
            if res.sclera_gains is not None else None,
            "base_l": round(base_lab.L, 3),
            "base_ab": [round(base_lab.a, 3), round(base_lab.b, 3)],
            "base_grade": _grade_of(base_lab.L, boundaries),
            "base_color": classify_iris_color(base_lab, config).code,
            "corr_l": round(res.lab.L, 3),
            "corr_ab": [round(res.lab.a, 3), round(res.lab.b, 3)],
            "corr_grade": int(res.grade.grade),
            "corr_color": res.iris_color.code,
        }
        records.append(record)
        if save_overlays:
            overlays = build_debug_images(
                res, eye_cfg, config=config, highlight_v=highlight_v
            )
            stem = f"{source['id']}__{Path(rel_text).stem}"
            for key in ("02_iris_ring", "06_sclera_samples"):
                if key in overlays:
                    _imwrite_unicode(OUT_DIR / f"{stem}__{key}.jpg", overlays[key])

    eligible_records = [r for r in records if r["eligible"]]
    analysis_rows = [
        {
            "id": r["id"],
            "operator": r["operator"],
            "group_name": r["group_name"],
            "subgroup_name": r["subgroup_name"],
            "camera_device": r["camera_device"],
            "light_device": r["light_device"],
            "illuminance": r["illuminance"],
            "lstar_before": r["base_l"],
            "lstar_after": r["corr_l"],
            "include_in_stats": True,
        }
        for r in eligible_records
    ]
    stability = analyze_stability(analysis_rows, boundaries)
    leave_group_out = []
    group_names = sorted({r["group_name"] for r in eligible_records if r["group_name"]})
    for held_out in group_names:
        fold_rows = [r for r in analysis_rows if r["group_name"] != held_out]
        fold = analyze_stability(fold_rows, boundaries)
        leave_group_out.append({
            "held_out_group": held_out,
            "n": len(fold_rows),
            "per_person_non_inferior": all(
                p["std_after"] <= p["std_before"] for p in fold.get("per_person", [])
            ),
            "per_person": fold.get("per_person", []),
        })
    density = {
        "n": len(eligible_records),
        "std_before": round(_sample_std([r["base_l"] for r in eligible_records]), 3),
        "std_after": round(_sample_std([r["corr_l"] for r in eligible_records]), 3),
    }
    acceptance = {
        "density_non_inferior": density["std_after"] <= density["std_before"],
        "per_person_non_inferior": all(
            p["std_after"] <= p["std_before"] for p in stability.get("per_person", [])
        ),
        "grade_majority_non_inferior": all(
            p["majority_after"] >= p["majority_before"]
            for p in stability.get("per_person", [])
        ),
        "within_median_non_inferior": (
            stability["within_subgroup"]["median_std_after"] is not None
            and stability["within_subgroup"]["median_std_after"]
            <= stability["within_subgroup"]["median_std_before"]
        ),
        "within_wins_not_less": (
            stability["within_subgroup"]["wins"] >= stability["within_subgroup"]["losses"]
        ),
        "cross_pair_wins_not_less": all(
            c["improved"] >= c["worsened"] for c in stability.get("cross_condition", [])
        ),
        "leave_group_out_passes": sum(
            int(f["per_person_non_inferior"]) for f in leave_group_out
        ),
        "leave_group_out_total": len(leave_group_out),
        "grade_changes": stability.get("boundary_effect", {}).get("grade_changed", 0),
    }
    status_counts = Counter(r["sclera_status"] for r in records)
    print(
        f"replayed={len(records)} eligible={len(eligible_records)} failures={len(failures)} "
        f"density σ {density['std_before']:.3f}->{density['std_after']:.3f}"
    )
    for person in stability.get("per_person", []):
        print(
            f"{person['operator']}: σ {person['std_before']:.3f}->{person['std_after']:.3f}, "
            f"grade majority {person['majority_before']:.1%}->{person['majority_after']:.1%}"
        )
    ws = stability["within_subgroup"]
    print(
        f"within groups: wins={ws['wins']} losses={ws['losses']} "
        f"median σ {ws['median_std_before']}->{ws['median_std_after']}"
    )
    print(f"status={dict(status_counts)} acceptance={acceptance}")
    out_json = OUT_DIR / f"summary_{tag}.json"
    out_json.write_text(
        json.dumps(
            {
                "source": "experiments",
                "config": config.get("sclera_normalization") or {},
                "density": density,
                "acceptance": acceptance,
                "stability": stability,
                "leave_group_out": leave_group_out,
                "records": records,
                "failures": failures,
                "status_counts": dict(status_counts),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"detail={out_json}")


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
    experiment_mode = False
    baseline_tolerance = 1.0
    target_l_override = None
    if "--experiments" in args:
        experiment_mode = True
        args.remove("--experiments")
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
    if "--baseline-tolerance" in args:
        i = args.index("--baseline-tolerance")
        baseline_tolerance = max(0.0, float(args[i + 1]))
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
    if experiment_mode:
        _run_experiment_eval(
            config,
            tag=tag,
            save_overlays=save_overlays,
            baseline_tolerance=baseline_tolerance,
        )
        return
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
