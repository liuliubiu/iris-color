"""弱亮度归一（normalize_luminance + luminance_strength）A/B 汇总。

优先复用 debug_output/_sclera_eval/summary_candidate_*.json 与
summary_final_device_profile_*.json 的已跑结果，避免重复全量回放。
结论写入 diagnose 同目录，供决定是否开启弱亮度项。

用法（在 iris-vision 目录）：
    python scripts/ab_weak_luminance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "debug_output" / "_sclera_eval"


def _summary_row(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cfg = payload.get("config") or {}
    if not isinstance(cfg, dict):
        return None
    acc = payload.get("acceptance") or {}
    stab = payload.get("stability") or {}
    pp = stab.get("per_person") or []
    within = stab.get("within_subgroup") or {}
    nl = cfg.get("normalize_luminance")
    ls = cfg.get("luminance_strength")
    if nl is None and ls is None and "device" not in str(cfg.get("algorithm_version") or ""):
        # 非巩膜候选摘要
        if not path.name.startswith("summary_candidate") and "device_profile" not in path.name:
            return None
    return {
        "file": path.name,
        "algorithm_version": cfg.get("algorithm_version"),
        "normalize_luminance": bool(nl) if nl is not None else None,
        "luminance_strength": ls,
        "preserve_iris_luminance_during_chroma": cfg.get(
            "preserve_iris_luminance_during_chroma"
        ),
        "per_person_non_inferior": acc.get("per_person_non_inferior"),
        "within_median_non_inferior": acc.get("within_median_non_inferior"),
        "grade_changed": (stab.get("boundary_effect") or {}).get("grade_changed"),
        "per_person_std": [
            {
                "operator": p.get("operator"),
                "std_before": p.get("std_before"),
                "std_after": p.get("std_after"),
                "improved": (p.get("std_after") or 0) < (p.get("std_before") or 0),
            }
            for p in pp
        ],
        "within_median_std_before": within.get("median_std_before"),
        "within_median_std_after": within.get("median_std_after"),
        "person_std_delta_sum": round(
            sum((p.get("std_before") or 0) - (p.get("std_after") or 0) for p in pp),
            4,
        ),
    }


def main() -> None:
    rows = []
    for path in sorted(OUT_DIR.glob("summary_*.json")):
        row = _summary_row(path)
        if row is not None:
            rows.append(row)

    weak = [
        r
        for r in rows
        if r.get("normalize_luminance") is True
        and float(r.get("luminance_strength") or 0.0) > 0
    ]
    off = [
        r
        for r in rows
        if r.get("normalize_luminance") is False
        or float(r.get("luminance_strength") or 0.0) == 0.0
    ]
    device = [r for r in rows if r.get("algorithm_version") and "device-profile" in str(r.get("algorithm_version"))]

    weak_pass = [r for r in weak if r.get("per_person_non_inferior") is True]
    weak_fail = [r for r in weak if r.get("per_person_non_inferior") is False]
    best_device = max(device, key=lambda r: r.get("person_std_delta_sum") or -999, default=None)
    best_weak = max(weak_pass, key=lambda r: r.get("person_std_delta_sum") or -999, default=None)

    evidence_supports_trial = (
        len(weak_pass) > 0
        and best_weak is not None
        and best_device is not None
        and (best_weak.get("person_std_delta_sum") or 0)
        > (best_device.get("person_std_delta_sum") or 0) + 0.05
    )

    report = {
        "n_summaries": len(rows),
        "n_weak_luminance_on": len(weak),
        "n_weak_pass_per_person": len(weak_pass),
        "n_weak_fail_per_person": len(weak_fail),
        "n_luminance_off_or_zero": len(off),
        "best_weak_pass": best_weak,
        "best_device_profile": best_device,
        "weak_fail_examples": weak_fail[:8],
        "recommendation": {
            "enable_weak_luminance": False,
            "reason": (
                "历史 candidate 在 luminance_strength>0 时多数破坏 per_person_non_inferior；"
                "当前最优为关闭亮度归一的 device-profile。"
                if not evidence_supports_trial
                else "存在弱亮度配置优于当前 device-profile，可进入小范围回放验证。"
            ),
            "keep_config": {
                "normalize_luminance": False,
                "luminance_strength": 0.0,
                "preserve_iris_luminance_during_chroma": True,
            },
            "evidence_supports_trial": evidence_supports_trial,
        },
        "rows": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "ab_weak_luminance.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "n_weak_on": len(weak),
        "n_weak_pass": len(weak_pass),
        "n_weak_fail": len(weak_fail),
        "recommendation": report["recommendation"],
        "best_device_profile": {
            "file": (best_device or {}).get("file"),
            "person_std_delta_sum": (best_device or {}).get("person_std_delta_sum"),
        },
    }, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
