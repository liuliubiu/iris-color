"""搜索设备级巩膜 L* 校正系数，目标为同人 L* 稳定且不改变 Grade。"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.experiment_stability import analyze_stability  # noqa: E402


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "debug_output" / "_sclera_eval" / "summary_final_adaptive_v2.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = [
        r for r in payload["records"]
        if r.get("eligible") and r.get("sclera_lab") and r.get("sclera_status") == "applied"
    ]
    devices = sorted({r["camera_device"] for r in records})
    targets = {
        device: float(np.median([
            r["sclera_lab"][0] for r in records if r["camera_device"] == device
        ]))
        for device in devices
    }
    by_id = {r["id"]: r for r in records}
    results = []
    honor_grid = np.arange(0.0, 0.65, 0.05)
    iphone_grid = np.arange(-0.85, 0.05, 0.05)
    caps = [1.0, 1.5, 2.0, 2.5, 3.0]
    quality_powers = [0.0, 0.5, 1.0]
    for honor_beta, iphone_beta, cap, quality_power in itertools.product(
        honor_grid, iphone_grid, caps, quality_powers
    ):
        betas = {"honor70": float(honor_beta), "iPhone 15 plus": float(iphone_beta)}
        rows = []
        deltas = []
        for r in records:
            beta = betas[r["camera_device"]]
            quality = r["sclera_quality"] ** quality_power if quality_power else 1.0
            delta = -beta * (r["sclera_lab"][0] - targets[r["camera_device"]]) * quality
            delta = float(np.clip(delta, -cap, cap))
            deltas.append(delta)
            rows.append({
                "id": r["id"],
                "operator": r["operator"],
                "group_name": r["group_name"],
                "subgroup_name": r["subgroup_name"],
                "camera_device": r["camera_device"],
                "light_device": r["light_device"],
                "illuminance": r["illuminance"],
                "lstar_before": r["base_l"],
                "lstar_after": r["base_l"] + delta,
            })
        report = analyze_stability(rows, payload["stability"]["boundaries"])
        people_ok = all(p["std_after"] < p["std_before"] for p in report["per_person"])
        within = report["within_subgroup"]
        within_ok = (
            within["median_std_after"] <= within["median_std_before"]
            and within["wins"] >= within["losses"]
        )
        grade_ok = report["boundary_effect"]["grade_changed"] == 0
        cross_ok = all(
            c["median_gap_after"] <= c["median_gap_before"]
            for c in report["cross_condition"]
        )
        if people_ok and within_ok and grade_ok:
            improvement = sum(
                p["std_before"] - p["std_after"] for p in report["per_person"]
            )
            results.append({
                "score": round(improvement, 4),
                "targets": {k: round(v, 3) for k, v in targets.items()},
                "betas": betas,
                "delta_cap": cap,
                "quality_power": quality_power,
                "delta_abs_median": round(float(np.median(np.abs(deltas))), 3),
                "delta_abs_p90": round(float(np.percentile(np.abs(deltas), 90)), 3),
                "per_person": report["per_person"],
                "within": {
                    "wins": within["wins"],
                    "losses": within["losses"],
                    "median_before": within["median_std_before"],
                    "median_after": within["median_std_after"],
                },
                "cross_condition": report["cross_condition"],
                "cross_non_inferior": cross_ok,
            })
    results.sort(
        key=lambda x: (x["score"], x["delta_abs_median"], x["within"]["wins"]),
        reverse=True,
    )
    print(json.dumps(results[:30], ensure_ascii=False, indent=2))
    print(f"matches={len(results)} records={len(records)} targets={targets}")

    # 对最佳斜率再搜索设备常量偏置；偏置只对齐设备均值，不改变设备内方差。
    offset_results = []
    fixed_betas = {"honor70": 0.35, "iPhone 15 plus": -0.30}
    for honor_offset, iphone_offset in itertools.product(
        np.arange(-1.0, 1.01, 0.1), np.arange(-1.0, 1.01, 0.1)
    ):
        offsets = {
            "honor70": float(honor_offset),
            "iPhone 15 plus": float(iphone_offset),
        }
        rows = []
        for r in records:
            device = r["camera_device"]
            delta = (
                -fixed_betas[device]
                * (r["sclera_lab"][0] - targets[device])
                * r["sclera_quality"]
                + offsets[device]
            )
            delta = float(np.clip(delta, -1.0, 1.0))
            rows.append({
                "id": r["id"],
                "operator": r["operator"],
                "group_name": r["group_name"],
                "subgroup_name": r["subgroup_name"],
                "camera_device": device,
                "light_device": r["light_device"],
                "illuminance": r["illuminance"],
                "lstar_before": r["base_l"],
                "lstar_after": r["base_l"] + delta,
            })
        report = analyze_stability(rows, payload["stability"]["boundaries"])
        within = report["within_subgroup"]
        if (
            all(p["std_after"] < p["std_before"] for p in report["per_person"])
            and within["median_std_after"] <= within["median_std_before"]
            and within["wins"] >= within["losses"]
            and report["boundary_effect"]["grade_changed"] == 0
            and all(
                c["median_gap_after"] <= c["median_gap_before"]
                for c in report["cross_condition"]
            )
        ):
            offset_results.append({
                "offsets": offsets,
                "per_person": report["per_person"],
                "within": {
                    "wins": within["wins"],
                    "losses": within["losses"],
                    "median_before": within["median_std_before"],
                    "median_after": within["median_std_after"],
                },
                "cross_condition": report["cross_condition"],
            })
    print("strict_offset_matches=")
    print(json.dumps(offset_results[:20], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
