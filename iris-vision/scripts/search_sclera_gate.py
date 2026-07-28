"""在一次实验回放结果上搜索巩膜质量门控，不重新执行图像定位。

用法：
    python scripts/search_sclera_gate.py debug_output/_sclera_eval/summary_candidate_y.json
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.experiment_stability import analyze_stability  # noqa: E402


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "debug_output" / "_sclera_eval" / "summary_candidate_y.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = [r for r in payload["records"] if r.get("eligible")]
    boundaries = payload["stability"]["boundaries"]
    results = []
    grid = itertools.product(
        [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.0, 0.03, 0.05, 0.08, 0.10, 0.15],
        [0.10, 0.15, 0.20, 0.30, 1.0],
        [0.0, 0.08, 0.10, 0.12, 0.15],
        [0.12, 0.15, 0.20, 0.25, 1.0],
    )
    for q_min, side_min, side_max, mad_min, mad_max in grid:
        rows = []
        applied = 0
        for r in records:
            use = (
                r["sclera_quality"] >= q_min
                and side_min <= r["side_luminance_gap"] <= side_max
                and mad_min <= r["luminance_mad_ratio"] <= mad_max
            )
            applied += int(use)
            rows.append({
                "id": r["id"],
                "operator": r["operator"],
                "group_name": r["group_name"],
                "subgroup_name": r["subgroup_name"],
                "camera_device": r["camera_device"],
                "light_device": r["light_device"],
                "illuminance": r["illuminance"],
                "lstar_before": r["base_l"],
                "lstar_after": r["corr_l"] if use else r["base_l"],
            })
        report = analyze_stability(rows, boundaries)
        people_ok = all(
            p["std_after"] <= p["std_before"] for p in report["per_person"]
        )
        within = report["within_subgroup"]
        within_ok = (
            within["median_std_after"] <= within["median_std_before"]
            and within["wins"] >= within["losses"]
        )
        cross_ok = all(
            c["median_gap_after"] < c["median_gap_before"]
            for c in report["cross_condition"]
        )
        if people_ok and within_ok and cross_ok:
            improvement = sum(
                p["std_before"] - p["std_after"] for p in report["per_person"]
            )
            results.append({
                "score": round(improvement, 4),
                "applied": applied,
                "quality_min": q_min,
                "side_min": side_min,
                "side_max": side_max,
                "mad_min": mad_min,
                "mad_max": mad_max,
                "wins": within["wins"],
                "losses": within["losses"],
                "median_before": within["median_std_before"],
                "median_after": within["median_std_after"],
                "people": report["per_person"],
                "cross_condition": report["cross_condition"],
            })
    results.sort(key=lambda x: (x["score"], x["applied"]), reverse=True)
    print(json.dumps(results[:20], ensure_ascii=False, indent=2))
    print(f"matches={len(results)}")


if __name__ == "__main__":
    main()
