"""诊断巩膜设备曲线对虹膜 L* 的实际位移与同人稳定性瓶颈。

默认读取最新 device-profile 评估摘要；也可传入任意 summary_*.json。

用法（在 iris-vision 目录）：
    python scripts/diagnose_sclera_lstar.py
    python scripts/diagnose_sclera_lstar.py debug_output/_sclera_eval/summary_final_device_profile_v3_offset.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.experiment_stability import analyze_stability  # noqa: E402

OUT_DIR = ROOT / "debug_output" / "_sclera_eval"
DEFAULT_SOURCES = [
    OUT_DIR / "summary_final_device_profile_v3_offset.json",
    OUT_DIR / "summary_final_device_profile_v3.json",
    OUT_DIR / "summary_final_adaptive_v2.json",
]


def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def _stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0, "lt1_ratio": 0.0}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "p90": round(_percentile(vals, 90), 4),
        "max": round(float(np.max(arr)), 4),
        "lt1_ratio": round(float(np.mean(arr < 1.0)), 4),
    }


def _device_targets(records: list[dict], profiles: dict) -> dict[str, float]:
    targets: dict[str, float] = {}
    for device in sorted({r["camera_device"] for r in records}):
        rs = [r for r in records if r["camera_device"] == device]
        profile_name = rs[0].get("camera_profile")
        cfg_t = None
        if profile_name and profile_name in profiles:
            cfg_t = profiles[profile_name].get("sclera_target_l")
        if cfg_t is not None:
            targets[device] = float(cfg_t)
        else:
            targets[device] = float(np.median([r["sclera_lab"][0] for r in rs]))
    return targets


def diagnose(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = payload.get("config") or {}
    profile_cfg = cfg.get("device_luminance_profiles") or {}
    profiles = profile_cfg.get("profiles") or {}
    records = [
        r
        for r in payload.get("records") or []
        if r.get("eligible")
        and r.get("sclera_lab")
        and r.get("sclera_status") == "applied"
    ]
    targets = _device_targets(records, profiles)

    requested = [abs(float(r.get("requested_lstar_delta") or 0.0)) for r in records]
    observed = [abs(float(r["corr_l"]) - float(r["base_l"])) for r in records]
    qualities = [float(r.get("sclera_quality") or 0.0) for r in records]
    sclera_absdev = [
        abs(float(r["sclera_lab"][0]) - targets[r["camera_device"]]) for r in records
    ]
    hit_cap = 0
    cap = float(profile_cfg.get("delta_lstar_max") or 0.0)
    if cap > 0:
        hit_cap = int(
            sum(
                abs(float(r.get("requested_lstar_delta") or 0.0)) >= cap - 1e-6
                for r in records
            )
        )

    per_device = {}
    for device in sorted(targets):
        rs = [r for r in records if r["camera_device"] == device]
        Ls = [float(r["sclera_lab"][0]) for r in rs]
        per_device[device] = {
            "n": len(rs),
            "sclera_l_median": round(float(np.median(Ls)), 3),
            "sclera_l_std": round(float(np.std(Ls, ddof=1)), 3) if len(Ls) > 1 else 0.0,
            "target_l": round(targets[device], 3),
            "absdev_median": round(_percentile([abs(x - targets[device]) for x in Ls], 50), 3),
            "absdev_p90": round(_percentile([abs(x - targets[device]) for x in Ls], 90), 3),
            "requested_abs_median": round(
                float(np.median([abs(float(r.get("requested_lstar_delta") or 0)) for r in rs])),
                3,
            ),
            "profile": rs[0].get("camera_profile") if rs else None,
            "beta": (
                profiles.get(rs[0].get("camera_profile") or "", {}).get("iris_l_per_sclera_l")
                if rs
                else None
            ),
            "offset": (
                profiles.get(rs[0].get("camera_profile") or "", {}).get("iris_lstar_offset")
                if rs
                else None
            ),
        }

    cross_device = []
    by_person: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_person[str(r.get("operator"))].append(r)
    for operator, rs in sorted(by_person.items()):
        by_dev: dict[str, list[float]] = defaultdict(list)
        by_dev_after: dict[str, list[float]] = defaultdict(list)
        for r in rs:
            by_dev[r["camera_device"]].append(float(r["base_l"]))
            by_dev_after[r["camera_device"]].append(float(r["corr_l"]))
        devices = sorted(by_dev)
        for i, a in enumerate(devices):
            for b in devices[i + 1 :]:
                gap_before = abs(float(np.median(by_dev[a])) - float(np.median(by_dev[b])))
                gap_after = abs(
                    float(np.median(by_dev_after[a])) - float(np.median(by_dev_after[b]))
                )
                cross_device.append({
                    "operator": operator,
                    "device_a": a,
                    "device_b": b,
                    "median_gap_before": round(gap_before, 3),
                    "median_gap_after": round(gap_after, 3),
                    "improved": gap_after < gap_before - 1e-9,
                })

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
        }
        for r in records
    ]
    boundaries = (payload.get("stability") or {}).get("boundaries") or [55, 45, 29, 19]
    stability = analyze_stability(analysis_rows, boundaries)

    # 瓶颈判定
    absdev_med = _percentile(sclera_absdev, 50)
    req_med = _percentile(requested, 50)
    betas = [
        abs(float(v.get("iris_l_per_sclera_l") or 0.0))
        for v in profiles.values()
    ]
    beta_med = float(np.median(betas)) if betas else 0.0
    cross_still_large = any(
        c["median_gap_after"] >= 0.8 for c in cross_device
    )
    cap_binding = cap > 0 and hit_cap >= max(1, int(0.05 * len(records)))
    bottleneck = "algorithm_weak_beta"
    if not records:
        bottleneck = "no_eligible_records"
    elif sum(1 for r in records if not r.get("camera_profile")) > 0.2 * len(records):
        bottleneck = "profile_match_miss"
    elif _percentile(qualities, 10) < 0.2:
        bottleneck = "low_sclera_quality"
    elif cap_binding and absdev_med >= 1.0:
        # 旧摘要 cap=1 时常触顶；同时巩膜偏差中位仍不大 → 需放宽 cap 并重拟合 offset/scale
        bottleneck = "delta_cap_binding_with_narrow_sclera_range"
    elif absdev_med < 1.5 and req_med < 1.0:
        bottleneck = "algorithm_weak_beta_and_narrow_sclera_range"
    elif cap_binding:
        bottleneck = "delta_cap_binding"

    report = {
        "source_n_records": len(payload.get("records") or []),
        "eligible_applied": len(records),
        "delta_lstar_max": cap,
        "cap_hit_count": hit_cap,
        "requested_abs_delta": _stats(requested),
        "observed_abs_delta": _stats(observed),
        "sclera_absdev_from_target": _stats(sclera_absdev),
        "quality": _stats(qualities),
        "profile_counts": dict(Counter(r.get("camera_profile") for r in records)),
        "device_counts": dict(Counter(r["camera_device"] for r in records)),
        "per_device": per_device,
        "cross_device_median_gap": cross_device,
        "cross_device_still_large": cross_still_large,
        "stability_summary": {
            "per_person": stability.get("per_person"),
            "within_subgroup": {
                "wins": stability.get("within_subgroup", {}).get("wins"),
                "losses": stability.get("within_subgroup", {}).get("losses"),
                "median_std_before": stability.get("within_subgroup", {}).get(
                    "median_std_before"
                ),
                "median_std_after": stability.get("within_subgroup", {}).get(
                    "median_std_after"
                ),
            },
            "cross_condition": stability.get("cross_condition"),
            "grade_changed": (stability.get("boundary_effect") or {}).get(
                "grade_changed"
            ),
        },
        "bottleneck": bottleneck,
        "conclusion": {
            "delta_lstar_max_is_binding": cap_binding,
            "typical_requested_delta_lt_1": req_med < 1.0,
            "main_cause": (
                "算法设计（小 β + 质量收缩 + 关闭亮度归一/色偏保光）；"
                "巩膜采样动态范围偏窄是次因。"
                if "weak_beta" in bottleneck
                else bottleneck
            ),
            "recommend_raise_cap_only": False,
            "recommend_refit_beta_offset": True,
            "recommend_cross_device_affine": cross_still_large,
            "recommend_weak_luminance_trial": False,
            "notes": [
                f"巩膜 |L-target| 中位 {absdev_med:.2f}，β≈{beta_med:.2f} 时理论 |ΔL*|≈{absdev_med * beta_med:.2f}",
                f"实际 requested |ΔL*| 中位 {req_med:.2f}；cap={cap} hit={hit_cap}/{len(records)}",
                "弱亮度归一历史 candidate 多数破坏 per_person_non_inferior，默认不建议开启",
            ],
        },
    }
    return report


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else next(
        (p for p in DEFAULT_SOURCES if p.exists()), None
    )
    if source is None:
        raise SystemExit("no summary json found; pass a path explicitly")
    payload = json.loads(source.read_text(encoding="utf-8"))
    report = diagnose(payload)
    report["source"] = str(source)
    out = OUT_DIR / "diagnose_sclera_lstar.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
